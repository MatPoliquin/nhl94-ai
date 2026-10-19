#include <emscripten/emscripten.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libretro.h"

#define MAX_WIDTH 720
#define MAX_HEIGHT 576
#define MAX_AUDIO_FRAMES 4096
#define MAX_VARIABLES 96

static uint8_t pixels[MAX_WIDTH * MAX_HEIGHT * 4];
static int16_t audio_samples[MAX_AUDIO_FRAMES * 2];
static unsigned video_width, video_height, sample_frames;
static unsigned pixel_format = RETRO_PIXEL_FORMAT_RGB565;
static uint16_t buttons[4];
static int multitap;
extern uint8_t io_reg[0x10];
static const struct retro_variable *variables;
static char variable_values[MAX_VARIABLES][128];
static struct retro_system_av_info av_info;
static int loaded, callback_error;

static void core_log(enum retro_log_level level, const char *format, ...)
{
    if (level < RETRO_LOG_WARN) return;
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

static bool environment(unsigned command, void *data)
{
    switch (command) {
        case RETRO_ENVIRONMENT_SET_VARIABLES: {
            variables = data;
            unsigned i;
            for (i = 0; i < MAX_VARIABLES && variables[i].key; i++) {
                const char *start = strchr(variables[i].value, ';');
                if (!start) return false;
                while (*++start == ' ') {}
                size_t length = strcspn(start, "|");
                if (length >= sizeof(variable_values[i])) return false;
                memcpy(variable_values[i], start, length);
                variable_values[i][length] = '\0';
            }
            return i < MAX_VARIABLES;
        }
        case RETRO_ENVIRONMENT_GET_VARIABLE: {
            struct retro_variable *variable = data;
            if (!strcmp(variable->key, "genesis_plus_gx_region_detect")) {
                variable->value = "ntsc-u";
                return true;
            }
            for (unsigned i = 0; variables && i < MAX_VARIABLES && variables[i].key; i++) {
                if (!strcmp(variable->key, variables[i].key)) {
                    variable->value = variable_values[i];
                    return true;
                }
            }
            variable->value = NULL;
            return false;
        }
        case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            *(bool *)data = false;
            return true;
        case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
            ((struct retro_log_callback *)data)->log = core_log;
            return true;
        case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
        case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            *(const char **)data = "/";
            return true;
        case RETRO_ENVIRONMENT_GET_CAN_DUPE:
            *(bool *)data = true;
            return true;
        case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            pixel_format = *(unsigned *)data;
            return pixel_format == RETRO_PIXEL_FORMAT_RGB565 ||
                   pixel_format == RETRO_PIXEL_FORMAT_0RGB1555;
        case RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO:
            av_info = *(struct retro_system_av_info *)data;
            return true;
        case RETRO_ENVIRONMENT_SET_GEOMETRY:
            av_info.geometry = *(struct retro_game_geometry *)data;
            return true;
        default:
            return false;
    }
}

static void video(const void *data, unsigned width, unsigned height, size_t pitch)
{
    if (!data) return;
    if (width > MAX_WIDTH || height > MAX_HEIGHT || pitch < width * 2) {
        callback_error = 1;
        return;
    }
    video_width = width;
    video_height = height;
    for (unsigned y = 0; y < height; y++) {
        const uint16_t *row = (const uint16_t *)((const uint8_t *)data + y * pitch);
        for (unsigned x = 0; x < width; x++) {
            unsigned pixel = row[x];
            unsigned red = pixel_format == RETRO_PIXEL_FORMAT_RGB565 ?
                (pixel >> 11) & 31 : (pixel >> 10) & 31;
            unsigned green = pixel_format == RETRO_PIXEL_FORMAT_RGB565 ?
                (pixel >> 5) & 63 : (pixel >> 5) & 31;
            unsigned green_max = pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? 63 : 31;
            size_t offset = (y * width + x) * 4;
            pixels[offset] = red * 255 / 31;
            pixels[offset + 1] = green * 255 / green_max;
            pixels[offset + 2] = (pixel & 31) * 255 / 31;
            pixels[offset + 3] = 255;
        }
    }
}

static size_t audio(const int16_t *data, size_t frames)
{
    if (frames > MAX_AUDIO_FRAMES) {
        callback_error = 1;
        return 0;
    }
    memcpy(audio_samples, data, frames * 2 * sizeof(int16_t));
    sample_frames = frames;
    return frames;
}

static void poll_input(void) {}

static int16_t read_input(unsigned port, unsigned device, unsigned index, unsigned id)
{
    (void)index;
    return port < 4 && device == RETRO_DEVICE_JOYPAD && id < 16 ?
        !!(buttons[port] & (1u << id)) : 0;
}

/* Keep the browser ABI numeric while the core uses the native LP64 save layout. */
EMSCRIPTEN_KEEPALIVE unsigned arena_alloc(unsigned size) { return (uintptr_t)malloc(size); }
EMSCRIPTEN_KEEPALIVE void arena_free(unsigned pointer) { free((void *)(uintptr_t)pointer); }

EMSCRIPTEN_KEEPALIVE int arena_load(unsigned pointer, unsigned size)
{
    const uint8_t *rom = (const uint8_t *)(uintptr_t)pointer;
    if (loaded || size < 512 || size > 4 * 1024 * 1024) return 0;
    FILE *file = fopen("/game.md", "wb");
    if (!file) return 0;
    size_t written = fwrite(rom, 1, size, file);
    int closed = fclose(file);
    if (written != size || closed) return 0;
    retro_set_environment(environment);
    retro_set_video_refresh(video);
    retro_set_audio_sample_batch(audio);
    retro_set_input_poll(poll_input);
    retro_set_input_state(read_input);
    retro_init();
    struct retro_game_info game = { .path = "/game.md" };
    if (!retro_load_game(&game)) return 0;
    loaded = 1;
    retro_set_controller_port_device(0, RETRO_DEVICE_SUBCLASS(RETRO_DEVICE_JOYPAD, 0));
    retro_set_controller_port_device(1, RETRO_DEVICE_SUBCLASS(RETRO_DEVICE_JOYPAD, 0));
    retro_get_system_av_info(&av_info);
    return 1;
}

EMSCRIPTEN_KEEPALIVE int arena_step(unsigned first, unsigned second, unsigned third, unsigned fourth)
{
    if (!loaded || first > 0xffff || second > 0xffff || third > 0xffff || fourth > 0xffff) return 0;
    buttons[0] = first;
    buttons[1] = second;
    buttons[2] = third;
    buttons[3] = fourth;
    sample_frames = 0;
    retro_run();
    return !callback_error;
}

static void set_multitap(int enabled)
{
    if (multitap == enabled) return;
    unsigned device = enabled ? RETRO_DEVICE_SUBCLASS(RETRO_DEVICE_JOYPAD, 3) :
        RETRO_DEVICE_SUBCLASS(RETRO_DEVICE_JOYPAD, 0);
    retro_set_controller_port_device(0, device);
    if (!enabled) retro_set_controller_port_device(1, RETRO_DEVICE_SUBCLASS(RETRO_DEVICE_JOYPAD, 0));
    multitap = enabled;
}

EMSCRIPTEN_KEEPALIVE unsigned arena_pixels(void) { return (uintptr_t)pixels; }
EMSCRIPTEN_KEEPALIVE unsigned arena_width(void) { return video_width; }
EMSCRIPTEN_KEEPALIVE unsigned arena_height(void) { return video_height; }
EMSCRIPTEN_KEEPALIVE unsigned arena_audio(void) { return (uintptr_t)audio_samples; }
EMSCRIPTEN_KEEPALIVE unsigned arena_audio_frames(void) { return sample_frames; }
EMSCRIPTEN_KEEPALIVE double arena_fps(void) { return av_info.timing.fps; }
EMSCRIPTEN_KEEPALIVE double arena_sample_rate(void) { return av_info.timing.sample_rate; }
EMSCRIPTEN_KEEPALIVE unsigned arena_ram(void)
{
    return (uintptr_t)retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
}
EMSCRIPTEN_KEEPALIVE unsigned arena_state_size(void) { return retro_serialize_size(); }
EMSCRIPTEN_KEEPALIVE int arena_save(unsigned pointer)
{
    void *state = (void *)(uintptr_t)pointer;
    memset(state, 0, retro_serialize_size());
    return loaded && retro_serialize(state, retro_serialize_size());
}
EMSCRIPTEN_KEEPALIVE int arena_restore(unsigned pointer)
{
    const void *state = (const void *)(uintptr_t)pointer;
    if (!loaded || !retro_unserialize(state, retro_serialize_size())) return 0;
    const uint8_t *ram = retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
    uint16_t adapter;
    memcpy(&adapter, ram + 0xd046, sizeof(adapter));
    set_multitap(adapter != 0);
    return 1;
}

static uint16_t ram_word(const uint8_t *ram, unsigned offset)
{
    uint16_t value;
    memcpy(&value, ram + offset, sizeof(value));
    return value;
}

static void set_ram_word(uint8_t *ram, unsigned offset, uint16_t value)
{
    memcpy(ram + offset, &value, sizeof(value));
}

static int selectable(const uint8_t *ram, int slot)
{
    return slot >= 0 && slot < 12 && slot % 6 != 5 &&
        (int16_t)ram_word(ram, 0xb04a + slot * 0x80 + 0x34) > 0;
}

EMSCRIPTEN_KEEPALIVE int arena_set_controllers(unsigned first, unsigned second, unsigned third, unsigned fourth)
{
    if (!loaded || first > 2 || second > 2 || third > 2 || fourth > 2) return 0;
    uint8_t *ram = retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM);
    if (!ram || retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM) != 65536) return 0;
    unsigned teams[4] = {first, second, third, fourth};
    int previous[4], selected[4] = {-1, -1, -1, -1}, used[12] = {0};
    int use_multitap = third || fourth;
    int changed = multitap != use_multitap;
    for (unsigned port = 0; port < 4; port++) {
        previous[port] = (int16_t)ram_word(ram, 0xc320 + port * 2);
        changed |= ram_word(ram, 0xc328 + port * 2) != teams[port];
    }
    if (!changed) return 1;
    if (ram_word(ram, 0xd048) != 0 ||
        ram_word(ram, 0xb04a + 5 * 0x80 + 0x34) != 0 ||
        ram_word(ram, 0xb04a + 11 * 0x80 + 0x34) != 0) return 0;

    for (unsigned port = 0; port < 4; port++) {
        int slot = previous[port];
        if (teams[port] && slot >= 0 && slot < 12 &&
            (unsigned)(slot / 6 + 1) == teams[port] && !used[slot]) {
            selected[port] = slot;
            used[slot] = 1;
        }
    }
    for (unsigned port = 0; port < 4; port++) {
        if (!teams[port] || selected[port] >= 0) continue;
        int base = (teams[port] - 1) * 6;
        int reference = previous[port] >= 0 ? previous[port] : previous[0];
        int preferred = base + (reference >= 0 && reference < 12 ? reference % 6 : 2);
        if (selectable(ram, preferred) && !used[preferred]) selected[port] = preferred;
        else {
            for (int slot = base; slot < base + 5; slot++) {
                if (selectable(ram, slot) && !used[slot]) {
                    selected[port] = slot;
                    break;
                }
            }
        }
        if (selected[port] < 0) return 0;
        used[selected[port]] = 1;
    }
    for (unsigned slot = 0; slot < 12; slot++) {
        unsigned offset = (0xb04a + slot * 0x80 + 0x62) ^ 1;
        ram[offset] = used[slot] ? ram[offset] | 8 : (ram[offset] & ~8) | 2;
    }
    for (unsigned port = 0; port < 4; port++) {
        set_ram_word(ram, 0xc328 + port * 2, teams[port]);
        set_ram_word(ram, 0xc320 + port * 2, (uint16_t)selected[port]);
    }
    if (multitap != use_multitap) {
        set_multitap(use_multitap);
        io_reg[5] = use_multitap ? 0x7f : 0x40;
    }
    set_ram_word(ram, 0xd046, use_multitap);
    unsigned mode = use_multitap ? 3 : first && second ? (first == second ? 3 : 4) : (first ? first : second);
    set_ram_word(ram, 0xd04a, mode);
    return 1;
}

EMSCRIPTEN_KEEPALIVE void arena_dispose(void)
{
    if (loaded) retro_unload_game();
    retro_deinit();
    loaded = 0;
    multitap = 0;
}
