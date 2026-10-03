# NHL '94 player ratings

Base ratings extracted directly from the **NHL Hockey '94** Genesis ROM in the local disassembly. All **634** roster records from the 26 NHL team rosters reached through the team-pointer table at ROM `0x00030E` are included. Rows are ordered by descending **neutral Overall**; ties use player name, then team.

These are the packed four-bit ROM ratings, not an in-game player-card snapshot. The ROM applies hot/cold adjustments at runtime, so displayed ratings and Overall can change. Names preserve the ROM text, including its spelling and truncation.

## Extraction and ranking

Each team pointer identifies a variable-length roster. A player record is a big-endian length word, a NUL-terminated name, and eight packed attribute bytes. `ROM` is the address of that record's length word, allowing every row to be checked against `nhl94.bin`.

The role mask at team-data offset `+0xA` identifies the leading goalie records. The ROM uses distinct Overall field masks and weights for skaters and goalies. With all hot/cold values set to zero, the values used to order these tables are:

```text
Skater Overall = Agl + 2*Spd + 3*Off + Def + 2*ShP + Chk + 3*Stk + 3*ShA + 2*End + Pass
Goalie Overall = Agl + Def + floor(9*Puck/2) + StkR + floor(9*StkL/2) + GlvR + GlvL
```

This follows `CalcAttrib`, `OvrPlayerWgtList`, and `OvrGoalWgtList` in the disassembly. See [the attribute-layout reference](nhl94%20Deep%20dive.md#7-how-player-attributes-become-gameplay-values) and [the goalie field mapping](nhl94%20Deep%20dive.md#94-which-goalie-attributes-map-where) for runtime semantics.

## Skaters

`#` is the BCD-decoded jersey number. `Hand` is decoded from the roster's low handedness bit (`L` for clear, `R` for set); the remaining bits in that nibble are retained fighting/injury metadata. `P/S` is the pass/shot bias.

| Rank | Overall | Team | Player | ROM | # | Wt | Agl | Spd | Off | Def | ShP | Chk | Hand | Stk | ShA | End | P/S | Pass | Agr |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | Pittsburgh | Mario Lemieux | `0x003704` | 66 | 10 | 5 | 4 | 6 | 4 | 4 | 3 | R | 6 | 6 | 6 | 0 | 6 | 2 |
| 2 | 94 | Boston | Ray Bourque | `0x000B72` | 77 | 10 | 5 | 4 | 4 | 6 | 5 | 6 | L | 5 | 5 | 6 | 2 | 5 | 2 |
| 3 | 94 | Detroit | Steve Yzerman | `0x0015F2` | 19 | 6 | 6 | 5 | 6 | 4 | 4 | 1 | R | 5 | 5 | 6 | 1 | 5 | 2 |
| 4 | 92 | Buffalo | Alexnder Mogilny | `0x000DDC` | 89 | 7 | 6 | 6 | 6 | 3 | 4 | 2 | L | 5 | 5 | 4 | 4 | 5 | 2 |
| 5 | 88 | Boston | Adam Oates | `0x000A20` | 12 | 7 | 5 | 4 | 6 | 5 | 3 | 3 | R | 5 | 4 | 5 | 0 | 6 | 1 |
| 6 | 88 | Chicago | Jeremy Roenick | `0x0012EE` | 27 | 4 | 5 | 5 | 4 | 4 | 5 | 2 | R | 5 | 5 | 5 | 2 | 5 | 3 |
| 7 | 88 | Buffalo | Pat LaFontaine | `0x000D1E` | 16 | 5 | 5 | 4 | 6 | 4 | 3 | 3 | R | 5 | 4 | 6 | 0 | 5 | 2 |
| 8 | 87 | Los Angeles | Luc Robitaille | `0x001F80` | 20 | 7 | 4 | 4 | 5 | 3 | 4 | 2 | L | 5 | 6 | 5 | 2 | 4 | 3 |
| 9 | 87 | Winnipeg | Teemu Selanne | `0x004CAE` | 13 | 6 | 5 | 6 | 5 | 3 | 4 | 3 | R | 4 | 5 | 5 | 4 | 4 | 2 |
| 10 | 85 | Toronto | Doug Gilmour | `0x0045EC` | 93 | 4 | 5 | 4 | 5 | 5 | 4 | 4 | L | 4 | 4 | 6 | 0 | 4 | 3 |
| 11 | 85 | Vancouver | Pavel Bure | `0x0049BE` | 10 | 5 | 5 | 6 | 5 | 4 | 4 | 2 | L | 5 | 4 | 4 | 4 | 4 | 2 |
| 12 | 84 | Boston | Cam Neely | `0x000AEE` | 8 | 10 | 4 | 4 | 5 | 4 | 4 | 4 | R | 4 | 5 | 5 | 3 | 4 | 4 |
| 13 | 83 | Philadelphia | Eric Lindros | `0x003428` | 88 | 12 | 4 | 3 | 4 | 4 | 4 | 5 | R | 4 | 6 | 5 | 2 | 4 | 4 |
| 14 | 83 | Calgary | Gary Roberts | `0x001054` | 10 | 7 | 4 | 4 | 5 | 4 | 4 | 4 | L | 4 | 5 | 5 | 1 | 3 | 4 |
| 15 | 83 | Quebec | Joe Sakic | `0x0039FC` | 19 | 6 | 4 | 4 | 5 | 4 | 4 | 2 | L | 4 | 5 | 5 | 2 | 5 | 2 |
| 16 | 82 | New York Rangers | Mike Gartner | `0x002EE8` | 22 | 7 | 5 | 5 | 4 | 4 | 5 | 2 | R | 5 | 3 | 5 | 5 | 5 | 2 |
| 17 | 82 | New York Islanders | Pierre Turgeon | `0x002B1A` | 77 | 9 | 4 | 4 | 5 | 3 | 4 | 3 | L | 4 | 5 | 5 | 1 | 4 | 1 |
| 18 | 82 | Chicago | Steve Larmer | `0x0013C4` | 28 | 7 | 4 | 4 | 4 | 6 | 4 | 4 | L | 4 | 4 | 6 | 3 | 4 | 2 |
| 19 | 81 | Detroit | Dino Ciccarelli | `0x0016DE` | 22 | 5 | 5 | 4 | 4 | 2 | 5 | 2 | R | 5 | 5 | 4 | 1 | 4 | 3 |
| 20 | 81 | Philadelphia | Mark Recchi | `0x003528` | 8 | 6 | 5 | 4 | 5 | 3 | 4 | 3 | L | 4 | 5 | 4 | 1 | 4 | 3 |
| 21 | 81 | Los Angeles | Wayne Gretzky | `0x001F26` | 99 | 4 | 6 | 4 | 5 | 4 | 2 | 2 | L | 6 | 2 | 6 | 0 | 6 | 0 |
| 22 | 80 | St. Louis | Brett Hull | `0x0040CA` | 16 | 9 | 4 | 4 | 5 | 3 | 6 | 3 | R | 5 | 3 | 4 | 4 | 3 | 2 |
| 23 | 80 | Detroit | Sergei Fedorov | `0x00160A` | 91 | 7 | 5 | 4 | 4 | 4 | 4 | 3 | L | 5 | 4 | 4 | 1 | 5 | 3 |
| 24 | 80 | Calgary | Theoren Fleury | `0x0010C4` | 14 | 3 | 5 | 5 | 4 | 5 | 4 | 4 | R | 4 | 3 | 6 | 1 | 3 | 3 |
| 25 | 79 | Quebec | Mats Sundin | `0x003AF6` | 13 | 7 | 4 | 4 | 5 | 4 | 4 | 1 | R | 4 | 5 | 4 | 0 | 4 | 3 |
| 26 | 79 | Los Angeles | Tomas Sandstrom | `0x001FF2` | 7 | 9 | 4 | 4 | 5 | 3 | 5 | 3 | L | 4 | 5 | 3 | 2 | 3 | 3 |
| 27 | 78 | Chicago | Chris Chelios | `0x001446` | 7 | 7 | 4 | 4 | 4 | 6 | 5 | 4 | R | 5 | 1 | 6 | 2 | 4 | 5 |
| 28 | 78 | Dallas | Mike Modano | `0x00220C` | 9 | 7 | 5 | 5 | 4 | 4 | 5 | 2 | L | 5 | 2 | 5 | 2 | 4 | 3 |
| 29 | 77 | St. Louis | Brendan Shanahan | `0x0040B0` | 19 | 10 | 3 | 3 | 5 | 4 | 4 | 2 | R | 4 | 5 | 4 | 2 | 4 | 4 |
| 30 | 77 | Washington | Dimitri Khristich | `0x004ED2` | 8 | 7 | 3 | 3 | 4 | 3 | 4 | 3 | R | 4 | 6 | 4 | 1 | 4 | 2 |
| 31 | 77 | New York Rangers | Mark Messier | `0x002E1E` | 11 | 10 | 5 | 4 | 4 | 4 | 3 | 5 | L | 5 | 3 | 4 | 0 | 5 | 3 |
| 32 | 77 | Detroit | Paul Coffey | `0x001752` | 77 | 9 | 6 | 5 | 4 | 3 | 4 | 2 | L | 6 | 1 | 5 | 1 | 5 | 3 |
| 33 | 77 | Winnipeg | Phil Housley | `0x004D1C` | 6 | 6 | 6 | 5 | 4 | 3 | 3 | 2 | L | 6 | 2 | 4 | 0 | 6 | 2 |
| 34 | 76 | New York Islanders | Benoit Hogue | `0x002B32` | 33 | 7 | 4 | 5 | 4 | 3 | 3 | 2 | L | 4 | 5 | 4 | 1 | 4 | 3 |
| 35 | 76 | Los Angeles | Jimmy Carson | `0x001F3E` | 12 | 9 | 4 | 4 | 4 | 3 | 4 | 2 | R | 4 | 5 | 4 | 2 | 4 | 1 |
| 36 | 76 | Pittsburgh | Kevin Stevens | `0x003764` | 25 | 11 | 3 | 4 | 5 | 3 | 3 | 3 | L | 4 | 4 | 5 | 3 | 4 | 4 |
| 37 | 76 | Montreal | Kirk Muller | `0x0024FE` | 11 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | L | 4 | 4 | 4 | 1 | 4 | 3 |
| 38 | 76 | Edmonton | Petr Klima | `0x001A1A` | 85 | 7 | 5 | 5 | 3 | 2 | 4 | 1 | R | 5 | 5 | 4 | 5 | 3 | 3 |
| 39 | 75 | Winnipeg | Alexei Zhamnov | `0x004BC8` | 10 | 7 | 5 | 3 | 4 | 3 | 3 | 3 | L | 5 | 4 | 4 | 1 | 5 | 2 |
| 40 | 75 | Vancouver | Trevor Linden | `0x0049D2` | 16 | 9 | 4 | 4 | 4 | 4 | 4 | 3 | R | 4 | 4 | 4 | 3 | 4 | 2 |
| 41 | 74 | Washington | Al Iafrate | `0x005016` | 34 | 11 | 4 | 4 | 4 | 4 | 6 | 4 | L | 4 | 2 | 4 | 4 | 4 | 4 |
| 42 | 74 | Calgary | Gary Suter | `0x001122` | 20 | 7 | 5 | 4 | 4 | 5 | 4 | 4 | L | 4 | 2 | 5 | 2 | 4 | 3 |
| 43 | 74 | Pittsburgh | Jaromir Jagr | `0x0037BC` | 68 | 10 | 5 | 4 | 4 | 3 | 3 | 4 | L | 5 | 3 | 4 | 1 | 4 | 2 |
| 44 | 74 | Washington | Peter Bondra | `0x004F8E` | 12 | 6 | 4 | 6 | 4 | 3 | 3 | 2 | L | 4 | 4 | 4 | 2 | 3 | 2 |
| 45 | 73 | Montreal | Brian Bellows | `0x0025D4` | 23 | 8 | 4 | 4 | 4 | 3 | 4 | 2 | R | 4 | 4 | 4 | 2 | 4 | 2 |
| 46 | 73 | St. Louis | Craig Janney | `0x003FEE` | 15 | 7 | 4 | 3 | 5 | 4 | 3 | 2 | L | 4 | 4 | 4 | 0 | 4 | 0 |
| 47 | 73 | Hartford | Pat Verbeek | `0x001D0E` | 16 | 7 | 3 | 4 | 4 | 3 | 4 | 3 | R | 4 | 4 | 4 | 2 | 4 | 4 |
| 48 | 73 | Philadelphia | Rod BrindAmour | `0x00343E` | 17 | 9 | 4 | 3 | 4 | 4 | 4 | 3 | L | 4 | 4 | 4 | 2 | 4 | 3 |
| 49 | 73 | Montreal | Stephan Lebeau | `0x002514` | 47 | 5 | 4 | 4 | 4 | 3 | 3 | 2 | R | 4 | 5 | 4 | 0 | 3 | 1 |
| 50 | 73 | Hartford | Zarley Zalapski | `0x001D7C` | 3 | 10 | 5 | 4 | 4 | 5 | 4 | 4 | L | 4 | 2 | 5 | 1 | 3 | 3 |
| 51 | 72 | Dallas | Dave Gagner | `0x002222` | 15 | 6 | 4 | 4 | 4 | 4 | 3 | 2 | L | 4 | 4 | 4 | 2 | 4 | 4 |
| 52 | 72 | Hartford | Geoff Sanderson | `0x001C94` | 8 | 6 | 4 | 4 | 4 | 3 | 4 | 2 | L | 4 | 4 | 4 | 3 | 3 | 1 |
| 53 | 72 | St. Louis | Jeff Brown | `0x004120` | 21 | 9 | 3 | 3 | 4 | 4 | 5 | 2 | R | 4 | 3 | 5 | 1 | 4 | 2 |
| 54 | 72 | Pittsburgh | Joe Mullen | `0x0037E8` | 7 | 6 | 4 | 3 | 4 | 3 | 3 | 3 | R | 4 | 5 | 4 | 2 | 3 | 0 |
| 55 | 72 | Calgary | Joe Nieuwendyk | `0x000FF6` | 25 | 8 | 4 | 4 | 4 | 3 | 4 | 3 | L | 4 | 4 | 4 | 3 | 2 | 2 |
| 56 | 72 | Quebec | Valeri Kamensky | `0x003A58` | 31 | 8 | 4 | 4 | 4 | 3 | 5 | 2 | R | 4 | 3 | 4 | 1 | 4 | 2 |
| 57 | 71 | Calgary | Al MacInnis | `0x001136` | 2 | 8 | 4 | 4 | 4 | 3 | 6 | 3 | R | 4 | 1 | 5 | 2 | 4 | 3 |
| 58 | 71 | Tampa Bay | Brian Bradley | `0x0042F0` | 19 | 4 | 4 | 3 | 4 | 3 | 4 | 3 | R | 3 | 5 | 4 | 2 | 3 | 3 |
| 59 | 71 | New York Rangers | Brian Leetch | `0x002F6C` | 2 | 6 | 6 | 3 | 4 | 4 | 4 | 2 | L | 5 | 1 | 5 | 2 | 5 | 2 |
| 60 | 71 | Vancouver | Cliff Ronning | `0x0048DA` | 7 | 5 | 5 | 5 | 4 | 3 | 2 | 2 | L | 4 | 3 | 5 | 1 | 4 | 1 |
| 61 | 71 | Toronto | Dave Andreychuk | `0x004660` | 14 | 11 | 3 | 3 | 4 | 4 | 4 | 4 | R | 3 | 4 | 5 | 4 | 3 | 2 |
| 62 | 71 | New York Rangers | Esa Tikkanen | `0x002E7C` | 10 | 9 | 5 | 5 | 3 | 4 | 4 | 5 | L | 4 | 2 | 4 | 4 | 4 | 3 |
| 63 | 71 | Washington | Mike Ridley | `0x004EBC` | 17 | 9 | 4 | 4 | 4 | 3 | 3 | 2 | L | 4 | 4 | 4 | 0 | 4 | 2 |
| 64 | 71 | Quebec | Owen Nolan | `0x003B0C` | 11 | 8 | 4 | 4 | 4 | 3 | 4 | 1 | R | 4 | 4 | 4 | 3 | 3 | 4 |
| 65 | 71 | Pittsburgh | Rick Tocchet | `0x0037D2` | 22 | 9 | 2 | 2 | 5 | 4 | 4 | 3 | R | 3 | 5 | 4 | 1 | 3 | 4 |
| 66 | 71 | Calgary | Robert Reichel | `0x00100E` | 26 | 6 | 4 | 4 | 4 | 3 | 4 | 2 | L | 4 | 4 | 3 | 2 | 4 | 2 |
| 67 | 71 | Pittsburgh | Ron Francis | `0x00371C` | 10 | 9 | 4 | 3 | 4 | 4 | 4 | 4 | L | 4 | 3 | 4 | 0 | 4 | 2 |
| 68 | 71 | New Jersey | Stephane Richer | `0x0028D8` | 44 | 9 | 4 | 4 | 4 | 3 | 5 | 2 | R | 4 | 3 | 4 | 4 | 3 | 2 |
| 69 | 71 | Winnipeg | Thomas Steen | `0x004BE0` | 25 | 8 | 4 | 3 | 4 | 3 | 3 | 3 | L | 4 | 4 | 4 | 0 | 5 | 3 |
| 70 | 71 | Montreal | Vincent Damphousse | `0x00255A` | 25 | 6 | 4 | 4 | 4 | 3 | 3 | 2 | L | 5 | 3 | 4 | 2 | 4 | 3 |
| 71 | 70 | Edmonton | Craig Simpson | `0x0019A2` | 18 | 8 | 3 | 3 | 4 | 3 | 4 | 2 | R | 4 | 5 | 3 | 1 | 3 | 2 |
| 72 | 70 | Montreal | Denis Savard | `0x00252C` | 18 | 5 | 5 | 4 | 4 | 3 | 3 | 2 | R | 5 | 3 | 3 | 0 | 4 | 3 |
| 73 | 70 | Boston | Joe Juneau | `0x000A78` | 49 | 5 | 4 | 4 | 4 | 3 | 4 | 2 | L | 4 | 3 | 4 | 0 | 4 | 1 |
| 74 | 70 | Pittsburgh | Larry Murphy | `0x003840` | 55 | 10 | 4 | 3 | 4 | 5 | 4 | 3 | R | 4 | 2 | 5 | 1 | 4 | 3 |
| 75 | 70 | Quebec | Mike Ricci | `0x003A10` | 9 | 7 | 4 | 4 | 4 | 3 | 3 | 3 | L | 3 | 5 | 3 | 0 | 4 | 3 |
| 76 | 70 | Toronto | Nikolai Borshevsky | `0x0046BA` | 16 | 6 | 4 | 4 | 4 | 2 | 3 | 2 | L | 4 | 4 | 4 | 2 | 4 | 1 |
| 77 | 70 | Detroit | Paul Ysebaert | `0x00166A` | 21 | 7 | 4 | 4 | 4 | 4 | 3 | 2 | L | 4 | 4 | 3 | 3 | 4 | 2 |
| 78 | 70 | Philadelphia | Pelle Eklund | `0x003456` | 9 | 5 | 5 | 4 | 4 | 3 | 3 | 3 | L | 4 | 3 | 4 | 0 | 4 | 1 |
| 79 | 70 | New Jersey | Scott Stevens | `0x002962` | 4 | 11 | 4 | 4 | 3 | 4 | 4 | 5 | L | 4 | 2 | 5 | 0 | 4 | 3 |
| 80 | 70 | Calgary | Sergei Makarov | `0x0010DC` | 42 | 6 | 5 | 4 | 4 | 2 | 2 | 1 | L | 5 | 4 | 3 | 0 | 5 | 2 |
| 81 | 69 | New Jersey | Alexnder Semak | `0x002804` | 20 | 6 | 4 | 3 | 4 | 4 | 3 | 3 | R | 4 | 4 | 3 | 2 | 4 | 2 |
| 82 | 69 | Vancouver | Greg Adams | `0x004968` | 8 | 7 | 3 | 3 | 4 | 4 | 3 | 3 | L | 3 | 5 | 4 | 1 | 3 | 1 |
| 83 | 69 | Toronto | John Cullen | `0x004602` | 19 | 7 | 4 | 3 | 4 | 3 | 3 | 2 | R | 4 | 4 | 4 | 1 | 4 | 3 |
| 84 | 69 | Washington | Kevin Hatcher | `0x004FFE` | 4 | 12 | 3 | 3 | 4 | 4 | 5 | 4 | R | 4 | 2 | 4 | 4 | 4 | 3 |
| 85 | 69 | Los Angeles | Tony Granato | `0x001F98` | 21 | 6 | 4 | 5 | 4 | 4 | 3 | 3 | R | 3 | 4 | 3 | 3 | 3 | 4 |
| 86 | 69 | Dallas | Ulf Dahlen | `0x0022DC` | 22 | 8 | 3 | 4 | 4 | 3 | 3 | 2 | L | 4 | 4 | 4 | 3 | 3 | 0 |
| 87 | 68 | Chicago | Brent Sutter | `0x001320` | 12 | 6 | 3 | 2 | 4 | 5 | 3 | 4 | R | 4 | 3 | 5 | 1 | 3 | 3 |
| 88 | 68 | Tampa Bay | Chris Kontos | `0x004308` | 16 | 8 | 3 | 3 | 4 | 3 | 3 | 3 | L | 3 | 5 | 4 | 3 | 3 | 0 |
| 89 | 68 | Edmonton | Dave Manson | `0x001A5C` | 24 | 9 | 4 | 4 | 3 | 5 | 5 | 4 | L | 4 | 1 | 5 | 4 | 3 | 4 |
| 90 | 68 | New York Islanders | Derek King | `0x002BBC` | 27 | 9 | 3 | 2 | 4 | 3 | 4 | 1 | L | 4 | 5 | 4 | 2 | 2 | 2 |
| 91 | 68 | Vancouver | Geoff Courtnall | `0x004936` | 14 | 7 | 5 | 5 | 4 | 3 | 3 | 1 | L | 4 | 3 | 3 | 2 | 4 | 4 |
| 92 | 68 | Toronto | Glenn Anderson | `0x0046D6` | 9 | 7 | 4 | 4 | 4 | 3 | 3 | 2 | L | 4 | 3 | 4 | 1 | 4 | 3 |
| 93 | 68 | Los Angeles | Jari Kurri | `0x00200C` | 17 | 8 | 4 | 3 | 4 | 4 | 4 | 2 | R | 4 | 3 | 4 | 1 | 3 | 2 |
| 94 | 68 | Chicago | Joe Murphy | `0x00141E` | 17 | 7 | 4 | 4 | 4 | 3 | 4 | 3 | L | 3 | 3 | 4 | 2 | 4 | 3 |
| 95 | 68 | Dallas | Russ Courtnall | `0x0022C4` | 26 | 6 | 5 | 6 | 4 | 2 | 4 | 2 | R | 3 | 3 | 3 | 4 | 3 | 2 |
| 96 | 68 | New York Rangers | Sergei Nemchinov | `0x002E34` | 13 | 6 | 4 | 3 | 3 | 5 | 3 | 4 | L | 3 | 4 | 5 | 2 | 3 | 2 |
| 97 | 68 | Quebec | Steve Duchesne | `0x003B50` | 28 | 8 | 4 | 4 | 4 | 4 | 4 | 2 | L | 4 | 2 | 4 | 1 | 4 | 2 |
| 98 | 68 | Toronto | Wendel Clark | `0x00467A` | 17 | 8 | 3 | 3 | 3 | 3 | 5 | 4 | L | 4 | 3 | 4 | 3 | 4 | 4 |
| 99 | 67 | Quebec | Andrei Kovalenko | `0x003B20` | 51 | 3 | 3 | 4 | 4 | 3 | 3 | 3 | L | 3 | 4 | 4 | 1 | 3 | 2 |
| 100 | 67 | New York Rangers | Darren Turcotte | `0x002E4E` | 8 | 6 | 4 | 4 | 4 | 3 | 3 | 2 | L | 4 | 3 | 4 | 4 | 3 | 2 |
| 101 | 67 | Boston | Dmitri Kvartalnov | `0x000A8C` | 10 | 6 | 4 | 4 | 4 | 2 | 3 | 2 | L | 4 | 3 | 4 | 2 | 4 | 1 |
| 102 | 67 | St. Louis | Nelson Emerson | `0x004004` | 7 | 4 | 4 | 4 | 4 | 3 | 3 | 2 | R | 4 | 3 | 4 | 1 | 3 | 2 |
| 103 | 67 | New York Islanders | Steve Thomas | `0x002BA6` | 32 | 6 | 4 | 4 | 4 | 3 | 3 | 4 | L | 3 | 3 | 4 | 2 | 4 | 3 |
| 104 | 66 | Buffalo | Dale Hawerchuk | `0x000D36` | 10 | 6 | 5 | 4 | 4 | 4 | 3 | 2 | L | 5 | 1 | 3 | 0 | 5 | 2 |
| 105 | 66 | Toronto | Dave Ellett | `0x004746` | 4 | 9 | 4 | 4 | 3 | 4 | 5 | 4 | L | 4 | 1 | 4 | 2 | 4 | 2 |
| 106 | 66 | Washington | Michal Pivonka | `0x004F04` | 20 | 8 | 3 | 3 | 4 | 3 | 3 | 3 | L | 4 | 3 | 4 | 0 | 4 | 3 |
| 107 | 66 | New Jersey | Peter Stastny | `0x002836` | 26 | 9 | 4 | 3 | 3 | 4 | 3 | 3 | L | 4 | 4 | 3 | 2 | 4 | 1 |
| 108 | 66 | Los Angeles | Rob Blake | `0x00204E` | 4 | 11 | 4 | 4 | 4 | 4 | 4 | 3 | R | 4 | 1 | 4 | 3 | 4 | 4 |
| 109 | 66 | Chicago | Steve Smith | `0x00145E` | 5 | 11 | 4 | 4 | 3 | 4 | 4 | 4 | L | 4 | 1 | 5 | 2 | 4 | 4 |
| 110 | 65 | Chicago | Christan Ruuttu | `0x001306` | 22 | 8 | 4 | 4 | 3 | 4 | 3 | 4 | L | 4 | 2 | 4 | 2 | 4 | 3 |
| 111 | 65 | Anaheim | Terry Yake | `0x00542C` | 25 | 6 | 3 | 3 | 4 | 3 | 3 | 3 | R | 3 | 4 | 4 | 0 | 3 | 2 |
| 112 | 65 | Hartford | Terry Yake | `0x001C34` | 25 | 6 | 3 | 3 | 4 | 3 | 3 | 3 | R | 3 | 4 | 4 | 0 | 3 | 2 |
| 113 | 65 | New Jersey | Valeri Zelepukin | `0x002864` | 25 | 6 | 4 | 4 | 4 | 3 | 3 | 1 | L | 4 | 3 | 3 | 1 | 4 | 3 |
| 114 | 64 | Buffalo | Bob Sweeney | `0x000D4E` | 20 | 9 | 4 | 2 | 3 | 5 | 3 | 4 | R | 3 | 4 | 4 | 2 | 3 | 3 |
| 115 | 64 | Chicago | Brian Noonan | `0x0013F0` | 10 | 7 | 3 | 3 | 3 | 3 | 4 | 3 | R | 4 | 3 | 4 | 5 | 3 | 3 |
| 116 | 64 | New Jersey | Claude Lemieux | `0x0028F2` | 22 | 11 | 4 | 4 | 4 | 3 | 4 | 3 | R | 3 | 2 | 4 | 3 | 3 | 4 |
| 117 | 64 | Boston | Don Sweeney | `0x000B88` | 32 | 4 | 4 | 4 | 3 | 4 | 4 | 4 | L | 4 | 1 | 4 | 1 | 4 | 2 |
| 118 | 64 | Boston | Glen Wesley | `0x000B9E` | 26 | 8 | 5 | 4 | 3 | 4 | 4 | 3 | L | 4 | 1 | 4 | 4 | 4 | 2 |
| 119 | 64 | New York Rangers | James Patrick | `0x002F82` | 3 | 9 | 4 | 4 | 3 | 4 | 4 | 4 | R | 4 | 1 | 4 | 2 | 4 | 3 |
| 120 | 64 | Calgary | Joel Otto | `0x001026` | 29 | 11 | 4 | 3 | 3 | 5 | 2 | 5 | R | 3 | 4 | 4 | 1 | 2 | 4 |
| 121 | 64 | Vancouver | Petr Nedved | `0x0048F2` | 19 | 5 | 2 | 2 | 4 | 3 | 2 | 1 | L | 3 | 6 | 4 | 2 | 3 | 3 |
| 122 | 64 | Edmonton | Shayne Corson | `0x00198A` | 9 | 9 | 4 | 4 | 3 | 4 | 3 | 4 | L | 4 | 2 | 4 | 2 | 3 | 4 |
| 123 | 64 | Boston | Vladimir Ruzicka | `0x000A4A` | 38 | 10 | 4 | 4 | 3 | 3 | 4 | 2 | L | 4 | 3 | 3 | 3 | 3 | 2 |
| 124 | 64 | Buffalo | Yuri Khmylev | `0x000D8E` | 13 | 7 | 4 | 3 | 3 | 4 | 4 | 3 | R | 3 | 4 | 3 | 3 | 3 | 2 |
| 125 | 63 | New York Rangers | Adam Graves | `0x002E92` | 9 | 6 | 4 | 4 | 4 | 3 | 2 | 3 | L | 3 | 3 | 4 | 5 | 3 | 4 |
| 126 | 63 | Hartford | Andrew Cassels | `0x001C1C` | 21 | 7 | 3 | 3 | 4 | 4 | 2 | 2 | L | 3 | 4 | 4 | 0 | 3 | 2 |
| 127 | 63 | Boston | Dave Poulin | `0x000A34` | 19 | 7 | 4 | 3 | 3 | 4 | 3 | 4 | L | 3 | 3 | 4 | 1 | 4 | 2 |
| 128 | 63 | New York Rangers | Ed Olczyk | `0x002E68` | 12 | 9 | 4 | 3 | 3 | 3 | 4 | 2 | L | 4 | 3 | 3 | 4 | 4 | 2 |
| 129 | 63 | Winnipeg | Evgeny Davydov | `0x004CC6` | 11 | 6 | 4 | 4 | 3 | 3 | 4 | 1 | R | 3 | 4 | 3 | 5 | 3 | 2 |
| 130 | 63 | San Jose | Kelly Kisio | `0x003D18` | 11 | 6 | 2 | 2 | 4 | 4 | 3 | 3 | R | 3 | 4 | 4 | 0 | 3 | 3 |
| 131 | 63 | Washington | Kelly Miller | `0x004FBA` | 10 | 8 | 4 | 4 | 3 | 3 | 3 | 2 | L | 4 | 3 | 3 | 2 | 4 | 1 |
| 132 | 63 | Vancouver | Murray Craven | `0x004950` | 32 | 6 | 3 | 3 | 4 | 3 | 3 | 3 | L | 3 | 4 | 3 | 0 | 3 | 1 |
| 133 | 63 | New York Islanders | Ray Ferraro | `0x002B76` | 20 | 6 | 3 | 3 | 3 | 3 | 4 | 2 | L | 4 | 3 | 4 | 3 | 3 | 3 |
| 134 | 63 | Detroit | Steve Chiasson | `0x001768` | 3 | 9 | 4 | 3 | 4 | 4 | 4 | 3 | L | 4 | 1 | 4 | 2 | 3 | 4 |
| 135 | 63 | New York Rangers | Tony Amonte | `0x002EFE` | 33 | 6 | 4 | 4 | 4 | 3 | 3 | 1 | R | 3 | 3 | 4 | 3 | 3 | 2 |
| 136 | 62 | Washington | Dale Hunter | `0x004EEE` | 32 | 8 | 2 | 2 | 4 | 4 | 3 | 4 | L | 2 | 4 | 4 | 0 | 4 | 4 |
| 137 | 62 | Detroit | Dallas Drake | `0x001622` | 28 | 4 | 4 | 4 | 3 | 4 | 3 | 2 | L | 3 | 3 | 4 | 1 | 3 | 3 |
| 138 | 62 | San Jose | Doug Wilson | `0x003E3E` | 24 | 7 | 4 | 3 | 3 | 3 | 6 | 3 | L | 4 | 1 | 3 | 3 | 4 | 3 |
| 139 | 62 | Montreal | Eric Desjardins | `0x00263C` | 28 | 9 | 4 | 3 | 3 | 4 | 3 | 4 | R | 4 | 2 | 4 | 2 | 3 | 3 |
| 140 | 62 | Winnipeg | Fredrik Olausson | `0x004D4A` | 4 | 9 | 4 | 3 | 4 | 3 | 4 | 1 | R | 4 | 2 | 3 | 1 | 4 | 1 |
| 141 | 62 | Philadelphia | Kevin Dineen | `0x00353E` | 11 | 7 | 4 | 5 | 3 | 4 | 2 | 2 | R | 3 | 3 | 4 | 5 | 3 | 4 |
| 142 | 62 | Dallas | Mark Tinordi | `0x002348` | 24 | 9 | 3 | 3 | 3 | 4 | 4 | 4 | L | 3 | 3 | 4 | 2 | 2 | 4 |
| 143 | 62 | Montreal | Matt Schneider | `0x002656` | 27 | 7 | 3 | 4 | 3 | 3 | 3 | 3 | L | 4 | 2 | 4 | 2 | 4 | 3 |
| 144 | 62 | Dallas | Neal Broten | `0x002238` | 7 | 4 | 4 | 4 | 3 | 5 | 3 | 3 | L | 3 | 2 | 4 | 3 | 4 | 1 |
| 145 | 62 | Calgary | Paul Ranheim | `0x00106A` | 28 | 8 | 4 | 4 | 3 | 4 | 3 | 1 | R | 4 | 3 | 3 | 4 | 3 | 1 |
| 146 | 62 | Winnipeg | Teppo Numminen | `0x004D32` | 27 | 7 | 4 | 3 | 3 | 5 | 3 | 3 | R | 4 | 2 | 4 | 1 | 3 | 2 |
| 147 | 61 | New Jersey | Bobby Holik | `0x002894` | 16 | 10 | 3 | 4 | 3 | 3 | 3 | 3 | R | 3 | 3 | 4 | 5 | 3 | 3 |
| 148 | 61 | Washington | Calle Johansson | `0x005040` | 6 | 9 | 4 | 3 | 3 | 4 | 4 | 3 | L | 4 | 1 | 4 | 1 | 4 | 2 |
| 149 | 61 | Los Angeles | Corey Millen | `0x001F54` | 23 | 4 | 3 | 4 | 4 | 3 | 2 | 1 | R | 3 | 4 | 3 | 3 | 3 | 3 |
| 150 | 61 | Winnipeg | Darrin Shannon | `0x004C3A` | 34 | 9 | 2 | 2 | 3 | 4 | 3 | 4 | L | 3 | 4 | 4 | 0 | 3 | 3 |
| 151 | 61 | Edmonton | Doug Weight | `0x001906` | 39 | 6 | 3 | 3 | 3 | 4 | 3 | 3 | L | 4 | 3 | 3 | 1 | 3 | 2 |
| 152 | 61 | Edmonton | Igor Kravchuk | `0x001A72` | 21 | 9 | 4 | 3 | 3 | 4 | 3 | 3 | L | 4 | 2 | 4 | 4 | 3 | 2 |
| 153 | 61 | New Jersey | John MacLean | `0x00287E` | 15 | 9 | 3 | 3 | 3 | 4 | 4 | 3 | R | 3 | 3 | 4 | 4 | 2 | 3 |
| 154 | 61 | St. Louis | Kevin Miller | `0x0040DE` | 14 | 7 | 4 | 3 | 3 | 4 | 3 | 3 | R | 3 | 3 | 4 | 3 | 3 | 3 |
| 155 | 61 | Detroit | Nicklas Lidstrom | `0x001796` | 5 | 5 | 4 | 3 | 3 | 4 | 4 | 3 | L | 4 | 1 | 4 | 2 | 4 | 1 |
| 156 | 61 | Washington | Pat Elynuik | `0x004FA4` | 19 | 6 | 3 | 3 | 3 | 3 | 3 | 2 | R | 3 | 5 | 3 | 1 | 2 | 2 |
| 157 | 61 | Toronto | Peter Zezel | `0x004632` | 25 | 9 | 3 | 3 | 3 | 3 | 3 | 4 | L | 3 | 3 | 4 | 1 | 4 | 1 |
| 158 | 61 | New York Islanders | Vladimir Malakhov | `0x002C6E` | 23 | 10 | 3 | 3 | 4 | 3 | 4 | 3 | L | 3 | 2 | 4 | 2 | 3 | 3 |
| 159 | 60 | Los Angeles | Alexei Zhitnik | `0x00207A` | 2 | 5 | 4 | 4 | 3 | 4 | 3 | 3 | L | 3 | 2 | 4 | 1 | 3 | 3 |
| 160 | 60 | Chicago | Dirk Graham | `0x0013DA` | 33 | 8 | 4 | 3 | 3 | 5 | 3 | 4 | R | 3 | 2 | 4 | 5 | 3 | 4 |
| 161 | 60 | Los Angeles | Mike Donnelly | `0x002020` | 11 | 6 | 4 | 4 | 4 | 3 | 2 | 3 | L | 2 | 3 | 4 | 3 | 3 | 2 |
| 162 | 60 | Toronto | Mike Krushelski | `0x004618` | 26 | 9 | 3 | 3 | 3 | 4 | 2 | 3 | L | 3 | 4 | 4 | 3 | 2 | 2 |
| 163 | 60 | St. Louis | Ron Sutter | `0x00401C` | 22 | 6 | 3 | 3 | 3 | 4 | 2 | 4 | R | 3 | 3 | 4 | 3 | 4 | 3 |
| 164 | 60 | Boston | Ted Donato | `0x000A64` | 21 | 4 | 3 | 4 | 3 | 4 | 3 | 3 | L | 3 | 3 | 3 | 3 | 3 | 2 |
| 165 | 59 | Quebec | Alexei Gusarov | `0x003B84` | 5 | 6 | 3 | 3 | 3 | 3 | 3 | 3 | L | 3 | 3 | 4 | 0 | 3 | 2 |
| 166 | 59 | New York Rangers | Alexei Kovalev | `0x002F14` | 27 | 7 | 4 | 3 | 3 | 3 | 3 | 1 | L | 3 | 4 | 3 | 4 | 3 | 3 |
| 167 | 59 | Anaheim | Anatoli Semenov | `0x005454` | 20 | 7 | 3 | 3 | 3 | 4 | 3 | 2 | L | 3 | 3 | 4 | 0 | 3 | 2 |
| 168 | 59 | Vancouver | Anatoli Semenov | `0x004908` | 20 | 7 | 3 | 3 | 3 | 4 | 3 | 2 | L | 3 | 3 | 4 | 0 | 3 | 2 |
| 169 | 59 | New Jersey | Bernie Nicholls | `0x00281C` | 19 | 6 | 3 | 3 | 4 | 3 | 3 | 1 | R | 4 | 2 | 3 | 0 | 4 | 3 |
| 170 | 59 | Ottawa | Bob Kudelski | `0x003226` | 26 | 9 | 3 | 3 | 3 | 3 | 3 | 2 | R | 3 | 4 | 3 | 4 | 3 | 2 |
| 171 | 59 | Hartford | Eric Weinrich | `0x001D96` | 4 | 10 | 3 | 4 | 3 | 4 | 3 | 3 | L | 3 | 2 | 4 | 1 | 3 | 3 |
| 172 | 59 | Hartford | Mark Janssens | `0x001D24` | 22 | 11 | 3 | 3 | 3 | 3 | 3 | 4 | L | 3 | 3 | 4 | 1 | 2 | 4 |
| 173 | 59 | Chicago | Michel Goulet | `0x00134C` | 16 | 8 | 4 | 3 | 3 | 3 | 2 | 2 | L | 3 | 5 | 3 | 3 | 1 | 2 |
| 174 | 59 | Dallas | Mike Craig | `0x0022F0` | 20 | 6 | 3 | 3 | 3 | 3 | 3 | 3 | R | 4 | 3 | 3 | 2 | 2 | 3 |
| 175 | 59 | San Jose | Pat Falloon | `0x003DFA` | 17 | 7 | 4 | 4 | 3 | 2 | 3 | 1 | R | 4 | 2 | 4 | 5 | 3 | 1 |
| 176 | 59 | New York Islanders | Patrick Flatley | `0x002BE4` | 26 | 8 | 3 | 2 | 4 | 3 | 2 | 3 | R | 4 | 2 | 4 | 0 | 4 | 2 |
| 177 | 59 | Detroit | Ray Sheppard | `0x0016F8` | 26 | 6 | 2 | 2 | 4 | 3 | 3 | 3 | R | 3 | 4 | 3 | 2 | 2 | 2 |
| 178 | 58 | Calgary | Brent Ashton | `0x001080` | 15 | 10 | 3 | 3 | 3 | 4 | 3 | 4 | L | 3 | 2 | 4 | 4 | 3 | 3 |
| 179 | 58 | New York Islanders | Brian Mullen | `0x002BFE` | 16 | 6 | 3 | 3 | 3 | 3 | 3 | 2 | L | 3 | 3 | 4 | 5 | 3 | 1 |
| 180 | 58 | Montreal | Gary Leeman | `0x002600` | 26 | 5 | 4 | 4 | 3 | 2 | 3 | 2 | R | 3 | 3 | 3 | 2 | 3 | 2 |
| 181 | 58 | Montreal | Gilbert Dionne | `0x002576` | 45 | 8 | 3 | 4 | 3 | 3 | 3 | 2 | L | 3 | 3 | 3 | 2 | 3 | 2 |
| 182 | 58 | Ottawa | Jamie Baker | `0x00311C` | 13 | 7 | 3 | 3 | 3 | 3 | 3 | 2 | L | 3 | 3 | 4 | 2 | 3 | 2 |
| 183 | 58 | Montreal | John Leclair | `0x00258E` | 17 | 11 | 3 | 4 | 3 | 3 | 3 | 3 | L | 3 | 3 | 3 | 2 | 2 | 2 |
| 184 | 58 | Detroit | John Ogrodnick | `0x0016C6` | 25 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | L | 3 | 4 | 2 | 1 | 3 | 0 |
| 185 | 58 | Winnipeg | Luciano Borsato | `0x004BF6` | 38 | 4 | 3 | 3 | 3 | 3 | 2 | 3 | R | 3 | 4 | 3 | 2 | 3 | 2 |
| 186 | 58 | Pittsburgh | Shawn McEachern | `0x003732` | 15 | 7 | 3 | 3 | 3 | 3 | 3 | 2 | L | 3 | 3 | 4 | 3 | 3 | 2 |
| 187 | 57 | Washington | Bob Carpenter | `0x004F4E` | 11 | 7 | 4 | 4 | 3 | 3 | 3 | 2 | L | 3 | 2 | 3 | 4 | 4 | 3 |
| 188 | 57 | Quebec | Curtis Leschyshyn | `0x003B68` | 7 | 9 | 3 | 4 | 3 | 4 | 2 | 3 | L | 2 | 3 | 4 | 0 | 3 | 2 |
| 189 | 57 | New York Islanders | Dave Volek | `0x002BD0` | 25 | 6 | 4 | 4 | 3 | 2 | 2 | 1 | L | 4 | 2 | 4 | 5 | 3 | 2 |
| 190 | 57 | Vancouver | Dixon Ward | `0x0049EA` | 17 | 9 | 2 | 3 | 3 | 3 | 3 | 2 | R | 2 | 5 | 3 | 1 | 2 | 3 |
| 191 | 57 | Vancouver | Jyrki Lumme | `0x004A28` | 21 | 7 | 4 | 3 | 3 | 4 | 2 | 3 | L | 4 | 1 | 4 | 1 | 4 | 2 |
| 192 | 57 | Los Angeles | Marty McSorley | `0x002062` | 33 | 14 | 3 | 3 | 3 | 4 | 3 | 3 | R | 3 | 2 | 4 | 4 | 3 | 6 |
| 193 | 57 | Montreal | Mike Keane | `0x0025EC` | 12 | 5 | 3 | 4 | 4 | 3 | 2 | 3 | R | 2 | 3 | 3 | 0 | 3 | 3 |
| 194 | 57 | Ottawa | Norm Maciver | `0x003280` | 22 | 6 | 3 | 3 | 4 | 4 | 3 | 3 | L | 2 | 2 | 4 | 1 | 3 | 3 |
| 195 | 57 | Buffalo | Petr Svoboda | `0x000E6A` | 7 | 5 | 4 | 4 | 3 | 4 | 3 | 3 | L | 4 | 0 | 4 | 0 | 3 | 3 |
| 196 | 57 | Quebec | Scott Young | `0x003B3A` | 48 | 7 | 3 | 3 | 3 | 4 | 3 | 2 | R | 3 | 3 | 3 | 4 | 3 | 1 |
| 197 | 57 | Vancouver | Sergio Momesso | `0x00497C` | 27 | 11 | 4 | 3 | 3 | 3 | 3 | 4 | L | 2 | 3 | 4 | 4 | 2 | 4 |
| 198 | 57 | Boston | Stephen Heinze | `0x000B1A` | 23 | 6 | 3 | 3 | 3 | 3 | 3 | 3 | R | 3 | 3 | 3 | 5 | 3 | 1 |
| 199 | 57 | Washington | Sylvain Cote | `0x00502A` | 3 | 6 | 3 | 3 | 3 | 4 | 3 | 3 | R | 3 | 2 | 4 | 4 | 3 | 2 |
| 200 | 57 | New York Islanders | Tom Kurvers | `0x002CD0` | 28 | 8 | 3 | 3 | 4 | 3 | 4 | 2 | L | 3 | 1 | 4 | 1 | 3 | 2 |
| 201 | 57 | Pittsburgh | Ulf Samuelsson | `0x003856` | 5 | 8 | 4 | 4 | 3 | 5 | 3 | 5 | L | 3 | 0 | 4 | 1 | 3 | 5 |
| 202 | 57 | Edmonton | Zdeno Ciger | `0x0019BA` | 8 | 7 | 4 | 3 | 3 | 3 | 3 | 2 | L | 3 | 3 | 3 | 2 | 3 | 0 |
| 203 | 56 | Hartford | Adam Burt | `0x001DAE` | 6 | 7 | 4 | 3 | 2 | 3 | 4 | 3 | L | 4 | 1 | 4 | 3 | 3 | 4 |
| 204 | 56 | Anaheim | Alexei Kasatonov | `0x0054AE` | 7 | 11 | 4 | 3 | 2 | 3 | 4 | 4 | L | 4 | 1 | 3 | 1 | 4 | 2 |
| 205 | 56 | New Jersey | Alexei Kasatonov | `0x0029C4` | 7 | 11 | 4 | 3 | 2 | 3 | 4 | 4 | L | 4 | 1 | 3 | 1 | 4 | 2 |
| 206 | 56 | Florida | Andrei Lomakin | `0x00521E` | 23 | 7 | 4 | 3 | 3 | 2 | 3 | 1 | L | 3 | 3 | 3 | 2 | 4 | 2 |
| 207 | 56 | Philadelphia | Andrei Lomakin | `0x0034E0` | 23 | 7 | 4 | 3 | 3 | 2 | 3 | 1 | L | 3 | 3 | 3 | 2 | 4 | 2 |
| 208 | 56 | Montreal | J.J. Daigneault | `0x0026B4` | 48 | 6 | 4 | 3 | 2 | 4 | 4 | 3 | L | 3 | 2 | 4 | 4 | 2 | 2 |
| 209 | 56 | Edmonton | Kevin Todd | `0x00194A` | 15 | 5 | 3 | 4 | 3 | 3 | 3 | 3 | L | 3 | 2 | 3 | 3 | 3 | 2 |
| 210 | 56 | Hartford | Patrick Poulin | `0x001CAE` | 24 | 10 | 3 | 3 | 3 | 2 | 3 | 3 | L | 3 | 3 | 3 | 2 | 3 | 2 |
| 211 | 56 | San Jose | Sandis Ozolinsh | `0x003E6C` | 6 | 7 | 3 | 3 | 3 | 3 | 3 | 3 | L | 3 | 2 | 4 | 2 | 3 | 3 |
| 212 | 56 | Boston | Stephen Leach | `0x000B02` | 27 | 6 | 3 | 3 | 3 | 3 | 3 | 3 | R | 3 | 2 | 4 | 5 | 3 | 3 |
| 213 | 56 | Detroit | Yves Racine | `0x001780` | 33 | 6 | 3 | 3 | 3 | 4 | 4 | 3 | L | 3 | 1 | 4 | 2 | 3 | 3 |
| 214 | 55 | Buffalo | Bob Errey | `0x000DB6` | 12 | 6 | 4 | 4 | 2 | 4 | 2 | 4 | L | 3 | 2 | 4 | 5 | 2 | 3 |
| 215 | 55 | Detroit | Bob Probert | `0x00170E` | 24 | 11 | 4 | 3 | 3 | 3 | 3 | 4 | L | 2 | 2 | 4 | 2 | 3 | 5 |
| 216 | 55 | Calgary | Brian Skrudland | `0x00103A` | 39 | 8 | 4 | 2 | 2 | 4 | 2 | 4 | L | 3 | 3 | 4 | 4 | 3 | 4 |
| 217 | 55 | Florida | Brian Skrudland | `0x0051DC` | 39 | 8 | 4 | 2 | 2 | 4 | 2 | 4 | L | 3 | 3 | 4 | 4 | 3 | 4 |
| 218 | 55 | New Jersey | Bruce Driver | `0x002994` | 23 | 6 | 3 | 3 | 3 | 4 | 2 | 3 | L | 3 | 2 | 4 | 2 | 3 | 2 |
| 219 | 55 | Edmonton | Craig MacTavish | `0x001930` | 14 | 8 | 4 | 3 | 2 | 4 | 2 | 4 | L | 3 | 2 | 5 | 2 | 2 | 3 |
| 220 | 55 | Boston | Dave Reid | `0x000AA8` | 17 | 9 | 2 | 3 | 3 | 3 | 3 | 1 | L | 2 | 4 | 4 | 4 | 2 | 0 |
| 221 | 55 | Buffalo | Donald Audette | `0x000E0E` | 28 | 5 | 3 | 3 | 3 | 3 | 3 | 2 | R | 3 | 3 | 3 | 5 | 2 | 3 |
| 222 | 55 | Vancouver | Doug Lidster | `0x004A3E` | 3 | 9 | 3 | 3 | 2 | 3 | 3 | 4 | R | 4 | 1 | 4 | 1 | 4 | 2 |
| 223 | 55 | Detroit | Gerard Gallant | `0x0016AE` | 17 | 6 | 2 | 2 | 3 | 3 | 3 | 4 | L | 3 | 3 | 3 | 1 | 3 | 4 |
| 224 | 55 | Boston | Gord Murphy | `0x000BC8` | 28 | 8 | 5 | 4 | 2 | 3 | 4 | 4 | R | 3 | 1 | 3 | 3 | 3 | 3 |
| 225 | 55 | Florida | Gord Murphy | `0x0052AC` | 28 | 8 | 5 | 4 | 2 | 3 | 4 | 4 | R | 3 | 1 | 3 | 3 | 3 | 3 |
| 226 | 55 | Toronto | Jamie Macoun | `0x004790` | 34 | 8 | 3 | 3 | 2 | 4 | 4 | 4 | L | 3 | 1 | 4 | 4 | 4 | 2 |
| 227 | 55 | Detroit | Keith Primeau | `0x001696` | 55 | 11 | 3 | 3 | 3 | 4 | 3 | 1 | L | 2 | 4 | 3 | 1 | 2 | 4 |
| 228 | 55 | Winnipeg | Keith Tkachuk | `0x004C52` | 7 | 9 | 3 | 2 | 3 | 3 | 4 | 3 | L | 2 | 3 | 4 | 4 | 2 | 4 |
| 229 | 55 | Boston | Peter Douris | `0x000B5C` | 16 | 8 | 3 | 4 | 3 | 3 | 3 | 2 | R | 3 | 3 | 2 | 4 | 2 | 1 |
| 230 | 55 | San Jose | Rob Gaudreau | `0x003D2E` | 37 | 6 | 3 | 3 | 3 | 3 | 3 | 2 | R | 2 | 3 | 4 | 5 | 3 | 1 |
| 231 | 55 | Hartford | Robert Kron | `0x001C62` | 38 | 5 | 3 | 3 | 3 | 4 | 2 | 2 | L | 3 | 3 | 3 | 4 | 3 | 2 |
| 232 | 55 | New York Rangers | Sergei Zubov | `0x002F9A` | 21 | 7 | 3 | 3 | 3 | 4 | 3 | 3 | R | 3 | 2 | 3 | 1 | 3 | 0 |
| 233 | 55 | Ottawa | Sylvain Turgeon | `0x0031B6` | 61 | 9 | 4 | 4 | 3 | 3 | 3 | 2 | L | 3 | 2 | 3 | 5 | 2 | 3 |
| 234 | 55 | Edmonton | Todd Elik | `0x00191C` | 34 | 7 | 3 | 4 | 3 | 3 | 2 | 2 | L | 3 | 3 | 3 | 1 | 2 | 3 |
| 235 | 55 | New Jersey | Vachslav Fetisov | `0x00297A` | 2 | 11 | 4 | 2 | 2 | 4 | 4 | 4 | L | 4 | 1 | 3 | 0 | 4 | 4 |
| 236 | 54 | Philadelphia | Dimitri Yushkevich | `0x00357E` | 2 | 7 | 4 | 3 | 3 | 4 | 3 | 2 | L | 3 | 1 | 4 | 3 | 3 | 3 |
| 237 | 54 | Philadelphia | Greg Hawgood | `0x00359A` | 20 | 7 | 4 | 3 | 3 | 3 | 3 | 2 | L | 3 | 2 | 3 | 1 | 3 | 3 |
| 238 | 54 | San Jose | Johan Garpenlov | `0x003D9E` | 10 | 6 | 3 | 3 | 4 | 2 | 2 | 1 | L | 3 | 3 | 3 | 1 | 2 | 2 |
| 239 | 54 | New York Rangers | Kevin Lowe | `0x002FC8` | 4 | 8 | 4 | 3 | 2 | 4 | 3 | 4 | L | 4 | 1 | 3 | 1 | 3 | 3 |
| 240 | 54 | Dallas | Mike McPhee | `0x00227E` | 17 | 9 | 4 | 3 | 3 | 3 | 3 | 4 | L | 2 | 2 | 4 | 4 | 2 | 2 |
| 241 | 54 | Quebec | Mikhail Tatarinov | `0x003BDE` | 4 | 8 | 3 | 3 | 2 | 2 | 5 | 4 | L | 4 | 1 | 2 | 4 | 4 | 3 |
| 242 | 54 | New Jersey | Scott Niedrmayer | `0x0029AA` | 27 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | L | 3 | 2 | 3 | 1 | 3 | 2 |
| 243 | 54 | Buffalo | Wayne Presley | `0x000DF6` | 18 | 6 | 3 | 3 | 3 | 3 | 4 | 2 | R | 2 | 3 | 3 | 3 | 2 | 3 |
| 244 | 53 | Philadelphia | Brent Fedyk | `0x0034B6` | 18 | 8 | 2 | 2 | 4 | 3 | 3 | 3 | R | 2 | 3 | 3 | 1 | 2 | 2 |
| 245 | 53 | Montreal | Guy Carbonneau | `0x002542` | 21 | 6 | 4 | 3 | 2 | 4 | 2 | 4 | R | 4 | 1 | 3 | 3 | 4 | 1 |
| 246 | 53 | New York Islanders | Jeff Norton | `0x002CA6` | 8 | 8 | 3 | 3 | 4 | 3 | 2 | 2 | L | 2 | 2 | 4 | 1 | 3 | 2 |
| 247 | 53 | Philadelphia | Josef Beranek | `0x00346C` | 42 | 6 | 3 | 3 | 3 | 3 | 2 | 1 | L | 3 | 3 | 3 | 4 | 3 | 3 |
| 248 | 53 | Calgary | Michel Petit | `0x001192` | 7 | 9 | 3 | 3 | 2 | 3 | 4 | 3 | R | 4 | 1 | 3 | 3 | 3 | 3 |
| 249 | 53 | Hartford | Mikael Nylander | `0x001C48` | 36 | 5 | 4 | 3 | 3 | 2 | 3 | 2 | L | 3 | 2 | 3 | 1 | 3 | 2 |
| 250 | 53 | Buffalo | Richard Smehlik | `0x000E80` | 42 | 10 | 3 | 3 | 3 | 4 | 3 | 4 | L | 3 | 1 | 3 | 0 | 3 | 2 |
| 251 | 53 | Florida | Stephane Richer | `0x0052EC` | 25 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | R | 3 | 2 | 3 | 3 | 2 | 2 |
| 252 | 53 | Dallas | Tommy Sjodin | `0x00235E` | 33 | 6 | 4 | 3 | 3 | 3 | 4 | 2 | R | 3 | 1 | 3 | 3 | 3 | 1 |
| 253 | 52 | Vancouver | Jiri Slegr | `0x004A6C` | 24 | 10 | 3 | 3 | 3 | 3 | 3 | 2 | L | 3 | 1 | 4 | 1 | 3 | 4 |
| 254 | 52 | Tampa Bay | John Tucker | `0x0043EE` | 14 | 9 | 2 | 3 | 3 | 3 | 3 | 2 | R | 3 | 2 | 3 | 2 | 3 | 3 |
| 255 | 52 | Philadelphia | Keith Acton | `0x003484` | 25 | 4 | 2 | 2 | 2 | 4 | 3 | 4 | L | 3 | 2 | 4 | 2 | 3 | 2 |
| 256 | 52 | Detroit | Mark Howe | `0x0017B0` | 4 | 6 | 3 | 3 | 3 | 4 | 3 | 3 | L | 3 | 1 | 3 | 0 | 3 | 1 |
| 257 | 52 | Quebec | Martin Rucinsky | `0x003A3E` | 25 | 5 | 3 | 2 | 3 | 2 | 3 | 1 | L | 3 | 3 | 3 | 1 | 3 | 2 |
| 258 | 52 | New York Islanders | Marty McInnis | `0x002B48` | 18 | 4 | 3 | 3 | 3 | 2 | 2 | 2 | R | 3 | 3 | 3 | 0 | 2 | 2 |
| 259 | 52 | Winnipeg | Mike Eagles | `0x004C10` | 36 | 7 | 3 | 3 | 2 | 4 | 2 | 5 | L | 2 | 2 | 5 | 1 | 2 | 3 |
| 260 | 52 | Detroit | Mike Sillinger | `0x001638` | 23 | 7 | 3 | 3 | 3 | 3 | 2 | 2 | R | 3 | 2 | 4 | 0 | 2 | 1 |
| 261 | 52 | Montreal | Patrice Brisebois | `0x00266E` | 43 | 5 | 3 | 3 | 3 | 3 | 3 | 3 | R | 2 | 2 | 4 | 3 | 2 | 3 |
| 262 | 52 | Winnipeg | Sergei Bautin | `0x004D64` | 3 | 6 | 3 | 3 | 2 | 4 | 3 | 4 | L | 3 | 1 | 4 | 2 | 3 | 3 |
| 263 | 52 | Detroit | Vachslav Kozlov | `0x001650` | 13 | 5 | 4 | 3 | 2 | 2 | 3 | 1 | L | 3 | 3 | 3 | 5 | 3 | 3 |
| 264 | 52 | Detroit | Vladimir Konstantov | `0x0017C4` | 16 | 6 | 3 | 3 | 2 | 4 | 3 | 4 | R | 3 | 1 | 4 | 2 | 3 | 3 |
| 265 | 51 | St. Louis | Bob Bassen | `0x004044` | 28 | 4 | 3 | 4 | 2 | 2 | 2 | 4 | L | 3 | 3 | 2 | 3 | 2 | 3 |
| 266 | 51 | Dallas | Brent Gilchrist | `0x00224E` | 41 | 6 | 4 | 3 | 2 | 2 | 3 | 2 | L | 4 | 2 | 2 | 5 | 3 | 2 |
| 267 | 51 | Edmonton | Brian Benning | `0x001A8A` | 19 | 8 | 2 | 2 | 3 | 4 | 3 | 3 | L | 2 | 2 | 4 | 2 | 3 | 4 |
| 268 | 51 | Dallas | Brian Propp | `0x0022AE` | 16 | 8 | 3 | 3 | 3 | 2 | 3 | 3 | L | 3 | 2 | 2 | 5 | 3 | 0 |
| 269 | 51 | Quebec | Claude Lapointe | `0x003A24` | 47 | 5 | 3 | 3 | 3 | 4 | 2 | 2 | L | 3 | 2 | 3 | 1 | 2 | 3 |
| 270 | 51 | Vancouver | Gerald Diduck | `0x004A80` | 4 | 10 | 3 | 3 | 2 | 3 | 4 | 4 | R | 3 | 1 | 3 | 4 | 3 | 4 |
| 271 | 51 | Chicago | Greg Gilbert | `0x00137E` | 14 | 7 | 2 | 2 | 3 | 3 | 2 | 4 | L | 3 | 3 | 3 | 1 | 1 | 2 |
| 272 | 51 | Calgary | Greg Paslawski | `0x0010F4` | 23 | 7 | 2 | 2 | 3 | 1 | 3 | 2 | R | 3 | 4 | 2 | 2 | 2 | 0 |
| 273 | 51 | New York Rangers | Jan Erixon | `0x002EBE` | 20 | 8 | 3 | 3 | 2 | 3 | 2 | 2 | L | 3 | 3 | 3 | 0 | 3 | 1 |
| 274 | 51 | Florida | Mike Hough | `0x0051F6` | 18 | 7 | 3 | 2 | 3 | 3 | 3 | 2 | L | 3 | 2 | 3 | 2 | 3 | 3 |
| 275 | 51 | Quebec | Mike Hough | `0x003A72` | 18 | 7 | 3 | 2 | 3 | 3 | 3 | 2 | L | 3 | 2 | 3 | 2 | 3 | 3 |
| 276 | 51 | New York Islanders | Scott Lachance | `0x002CE6` | 7 | 8 | 2 | 2 | 2 | 4 | 3 | 3 | L | 3 | 2 | 4 | 1 | 3 | 3 |
| 277 | 51 | Toronto | Todd Gill | `0x004732` | 23 | 6 | 3 | 3 | 3 | 3 | 2 | 4 | L | 2 | 2 | 4 | 1 | 2 | 3 |
| 278 | 51 | New York Islanders | Uwe Krupp | `0x002CBC` | 4 | 14 | 2 | 2 | 3 | 4 | 3 | 3 | R | 2 | 2 | 4 | 1 | 3 | 2 |
| 279 | 50 | Calgary | Chris Lindberg | `0x001096` | 11 | 7 | 3 | 4 | 2 | 3 | 2 | 2 | L | 3 | 3 | 2 | 3 | 2 | 1 |
| 280 | 50 | New York Islanders | Darius Kasparitis | `0x002C8A` | 11 | 7 | 4 | 3 | 2 | 3 | 3 | 4 | L | 3 | 1 | 3 | 2 | 3 | 4 |
| 281 | 50 | New Jersey | Dave Barr | `0x00294E` | 11 | 8 | 3 | 3 | 2 | 3 | 3 | 3 | R | 2 | 3 | 3 | 2 | 2 | 2 |
| 282 | 50 | Toronto | Dimitri Mironov | `0x00475C` | 15 | 7 | 3 | 2 | 3 | 3 | 3 | 1 | L | 3 | 2 | 3 | 1 | 3 | 2 |
| 283 | 50 | St. Louis | Doug Crossman | `0x00414C` | 6 | 7 | 2 | 2 | 3 | 3 | 2 | 2 | L | 2 | 3 | 4 | 0 | 3 | 2 |
| 284 | 50 | Boston | Gregori Pantaleyev | `0x000ABC` | 13 | 6 | 4 | 4 | 2 | 2 | 2 | 1 | L | 3 | 3 | 2 | 4 | 3 | 1 |
| 285 | 50 | Chicago | Jocelyn Lemieux | `0x001394` | 26 | 9 | 3 | 4 | 3 | 3 | 4 | 2 | L | 2 | 2 | 2 | 2 | 1 | 3 |
| 286 | 50 | Quebec | Kerry Huffman | `0x003B9C` | 2 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | L | 3 | 1 | 3 | 2 | 2 | 3 |
| 287 | 50 | Tampa Bay | Marc Bureau | `0x004336` | 28 | 7 | 3 | 3 | 3 | 3 | 2 | 2 | R | 3 | 2 | 3 | 3 | 2 | 4 |
| 288 | 50 | Tampa Bay | Mikael Andersson | `0x0043A6` | 34 | 6 | 2 | 4 | 2 | 3 | 2 | 1 | L | 3 | 2 | 4 | 5 | 3 | 0 |
| 289 | 50 | Toronto | Mike Foligno | `0x004704` | 71 | 8 | 2 | 2 | 2 | 3 | 2 | 5 | R | 3 | 3 | 3 | 5 | 2 | 3 |
| 290 | 50 | New York Rangers | Phil Bourque | `0x002EA8` | 29 | 8 | 2 | 4 | 2 | 3 | 2 | 4 | L | 3 | 2 | 3 | 2 | 2 | 2 |
| 291 | 50 | Buffalo | Randy Wood | `0x000D7A` | 19 | 8 | 1 | 5 | 3 | 3 | 2 | 3 | L | 2 | 2 | 3 | 4 | 2 | 3 |
| 292 | 50 | Detroit | Sheldon Kennedy | `0x001724` | 15 | 4 | 3 | 3 | 3 | 2 | 2 | 2 | R | 2 | 4 | 2 | 5 | 2 | 2 |
| 293 | 50 | Tampa Bay | Steve Kasper | `0x004360` | 11 | 5 | 3 | 3 | 1 | 3 | 2 | 4 | L | 3 | 3 | 3 | 2 | 3 | 1 |
| 294 | 49 | New Jersey | Bill Guerin | `0x00290A` | 12 | 9 | 2 | 2 | 3 | 2 | 3 | 2 | R | 3 | 3 | 2 | 3 | 2 | 3 |
| 295 | 49 | Dallas | Bobby Smith | `0x002268` | 18 | 10 | 3 | 3 | 2 | 3 | 2 | 2 | L | 3 | 2 | 3 | 4 | 4 | 0 |
| 296 | 49 | Edmonton | Brad Werenka | `0x001AFE` | 36 | 9 | 3 | 3 | 2 | 2 | 3 | 2 | L | 3 | 3 | 2 | 5 | 2 | 3 |
| 297 | 49 | Boston | C.J. Young | `0x000B32` | 18 | 6 | 3 | 3 | 2 | 2 | 3 | 2 | R | 3 | 3 | 2 | 3 | 2 | 2 |
| 298 | 49 | Dallas | Gaetan Duchesne | `0x002294` | 10 | 9 | 3 | 4 | 2 | 3 | 2 | 3 | L | 2 | 3 | 3 | 5 | 1 | 1 |
| 299 | 49 | St. Louis | Garth Butcher | `0x004134` | 5 | 9 | 2 | 3 | 2 | 4 | 2 | 4 | R | 3 | 1 | 4 | 4 | 3 | 4 |
| 300 | 49 | Dallas | Jim Johnson | `0x002374` | 6 | 7 | 2 | 3 | 2 | 4 | 2 | 3 | L | 3 | 1 | 4 | 1 | 4 | 3 |
| 301 | 49 | Ottawa | Mark Freer | `0x00315A` | 11 | 6 | 2 | 2 | 3 | 3 | 2 | 2 | L | 2 | 3 | 4 | 3 | 2 | 2 |
| 302 | 49 | Washington | Paul MacDermid | `0x004FE6` | 23 | 9 | 2 | 2 | 2 | 3 | 2 | 4 | R | 2 | 4 | 3 | 3 | 2 | 3 |
| 303 | 49 | Tampa Bay | Rob Zamuner | `0x0043C0` | 7 | 9 | 2 | 3 | 3 | 2 | 2 | 3 | L | 3 | 2 | 3 | 3 | 2 | 2 |
| 304 | 49 | Winnipeg | Stu Barnes | `0x004C26` | 14 | 5 | 2 | 3 | 3 | 3 | 2 | 2 | R | 2 | 3 | 3 | 4 | 2 | 1 |
| 305 | 49 | Chicago | Troy Murray | `0x001336` | 19 | 8 | 4 | 3 | 2 | 4 | 2 | 4 | R | 3 | 1 | 3 | 5 | 3 | 3 |
| 306 | 48 | Ottawa | Brad Shaw | `0x003296` | 4 | 7 | 3 | 2 | 3 | 3 | 3 | 2 | R | 3 | 1 | 3 | 2 | 3 | 2 |
| 307 | 48 | Edmonton | Brian Glynn | `0x001AB8` | 6 | 11 | 3 | 3 | 2 | 3 | 4 | 3 | L | 2 | 1 | 4 | 3 | 2 | 3 |
| 308 | 48 | Vancouver | Dana Murzyn | `0x004AAE` | 5 | 9 | 2 | 3 | 2 | 3 | 4 | 4 | L | 2 | 1 | 4 | 4 | 2 | 4 |
| 309 | 48 | Los Angeles | Darryl Sydor | `0x002092` | 25 | 9 | 3 | 3 | 2 | 3 | 3 | 3 | L | 3 | 1 | 3 | 2 | 3 | 2 |
| 310 | 48 | Toronto | Dave McLlwain | `0x004648` | 7 | 7 | 3 | 2 | 2 | 3 | 1 | 3 | L | 3 | 3 | 3 | 5 | 3 | 2 |
| 311 | 48 | Los Angeles | Dave Taylor | `0x002038` | 18 | 7 | 3 | 2 | 2 | 3 | 3 | 3 | R | 3 | 2 | 3 | 3 | 2 | 3 |
| 312 | 48 | Buffalo | Doug Bodger | `0x000E54` | 8 | 10 | 2 | 3 | 3 | 4 | 2 | 3 | L | 2 | 1 | 4 | 1 | 3 | 3 |
| 313 | 48 | Winnipeg | Kris King | `0x004C6A` | 17 | 10 | 3 | 3 | 2 | 4 | 2 | 3 | L | 2 | 2 | 4 | 3 | 2 | 4 |
| 314 | 48 | Ottawa | Laurie Boschman | `0x00316E` | 16 | 6 | 4 | 2 | 2 | 4 | 2 | 3 | L | 3 | 2 | 3 | 5 | 2 | 3 |
| 315 | 48 | Ottawa | Mark Lamb | `0x003132` | 7 | 6 | 4 | 3 | 2 | 4 | 2 | 3 | L | 3 | 1 | 3 | 3 | 3 | 3 |
| 316 | 48 | Washington | Paul Cavallini | `0x00505A` | 14 | 10 | 3 | 3 | 2 | 3 | 3 | 3 | L | 3 | 1 | 3 | 5 | 3 | 2 |
| 317 | 48 | Pittsburgh | Paul Stanton | `0x003880` | 23 | 8 | 3 | 3 | 2 | 3 | 4 | 3 | R | 2 | 1 | 3 | 5 | 4 | 3 |
| 318 | 48 | Chicago | Rob Brown | `0x001432` | 44 | 6 | 4 | 3 | 3 | 2 | 2 | 2 | L | 4 | 1 | 2 | 0 | 2 | 4 |
| 319 | 48 | Edmonton | Scott Mellanby | `0x001A2E` | 27 | 9 | 1 | 1 | 3 | 3 | 3 | 4 | R | 2 | 3 | 3 | 3 | 2 | 4 |
| 320 | 48 | Florida | Scott Mellanby | `0x0051C4` | 27 | 9 | 1 | 1 | 3 | 3 | 3 | 4 | R | 2 | 3 | 3 | 3 | 2 | 4 |
| 321 | 47 | Toronto | Bill Berg | `0x0046A6` | 10 | 7 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 3 | 4 | 5 | 2 | 3 |
| 322 | 47 | Tampa Bay | Bob Beers | `0x004446` | 2 | 9 | 3 | 3 | 3 | 3 | 2 | 2 | R | 2 | 2 | 3 | 3 | 2 | 3 |
| 323 | 47 | Winnipeg | Bryan Erickson | `0x004CF2` | 18 | 5 | 2 | 3 | 3 | 3 | 2 | 3 | R | 2 | 2 | 3 | 1 | 2 | 1 |
| 324 | 47 | Los Angeles | Charlie Huddy | `0x0020A8` | 22 | 10 | 3 | 2 | 2 | 4 | 3 | 3 | L | 3 | 1 | 3 | 1 | 3 | 2 |
| 325 | 47 | Buffalo | Colin Patterson | `0x000E3A` | 17 | 8 | 4 | 4 | 2 | 3 | 1 | 3 | R | 2 | 3 | 2 | 5 | 2 | 2 |
| 326 | 47 | Tampa Bay | Danton Cole | `0x004404` | 24 | 7 | 3 | 3 | 3 | 3 | 2 | 2 | R | 2 | 2 | 3 | 4 | 2 | 1 |
| 327 | 47 | Vancouver | Dave Babych | `0x004A98` | 44 | 11 | 2 | 2 | 3 | 4 | 3 | 2 | L | 3 | 0 | 4 | 2 | 3 | 3 |
| 328 | 47 | Chicago | Dave Christian | `0x001406` | 25 | 8 | 3 | 3 | 2 | 3 | 3 | 2 | R | 3 | 1 | 3 | 2 | 3 | 1 |
| 329 | 47 | Philadelphia | Garry Galley | `0x003568` | 3 | 7 | 2 | 2 | 3 | 4 | 2 | 3 | L | 3 | 1 | 3 | 2 | 3 | 3 |
| 330 | 47 | Ottawa | Jeff Lazaro | `0x0031FA` | 28 | 6 | 3 | 3 | 3 | 1 | 2 | 2 | L | 3 | 3 | 1 | 5 | 2 | 2 |
| 331 | 47 | Philadelphia | Ric Nattress | `0x0035C8` | 5 | 10 | 2 | 2 | 3 | 3 | 2 | 2 | R | 2 | 3 | 3 | 3 | 2 | 2 |
| 332 | 47 | Montreal | Rob Ramage | `0x0026A0` | 5 | 9 | 3 | 2 | 2 | 3 | 3 | 4 | R | 3 | 1 | 3 | 5 | 3 | 4 |
| 333 | 47 | Hartford | Robert Petrovicky | `0x001C78` | 39 | 5 | 4 | 3 | 2 | 2 | 3 | 2 | L | 3 | 1 | 3 | 4 | 3 | 3 |
| 334 | 47 | Quebec | Scott Pearson | `0x003A9E` | 22 | 9 | 2 | 2 | 2 | 2 | 2 | 3 | L | 2 | 4 | 3 | 5 | 2 | 4 |
| 335 | 47 | New Jersey | Scott Pellerin | `0x002936` | 18 | 6 | 2 | 3 | 3 | 3 | 2 | 2 | L | 2 | 3 | 2 | 2 | 2 | 3 |
| 336 | 47 | Winnipeg | Tie Domi | `0x004D0A` | 20 | 9 | 2 | 2 | 2 | 3 | 2 | 3 | R | 2 | 3 | 4 | 1 | 2 | 6 |
| 337 | 47 | Vancouver | Tom Fergus | `0x004922` | 15 | 10 | 2 | 2 | 3 | 3 | 2 | 2 | L | 2 | 3 | 3 | 0 | 2 | 2 |
| 338 | 46 | Tampa Bay | Adam Creighton | `0x00431E` | 10 | 10 | 3 | 2 | 3 | 2 | 2 | 3 | L | 2 | 2 | 3 | 4 | 3 | 3 |
| 339 | 46 | Boston | Glen Feathrston | `0x000BF6` | 6 | 11 | 3 | 3 | 2 | 2 | 2 | 4 | L | 2 | 3 | 2 | 3 | 2 | 4 |
| 340 | 46 | Tampa Bay | Jason Lafreniere | `0x004376` | 17 | 6 | 2 | 2 | 3 | 3 | 2 | 1 | R | 2 | 4 | 2 | 2 | 1 | 2 |
| 341 | 46 | Calgary | Kevin Dahl | `0x0011A8` | 4 | 7 | 2 | 3 | 2 | 4 | 2 | 3 | R | 3 | 1 | 3 | 1 | 3 | 2 |
| 342 | 46 | Pittsburgh | Kjell Samuelsson | `0x0038C6` | 28 | 14 | 2 | 2 | 1 | 3 | 4 | 4 | R | 3 | 1 | 4 | 5 | 2 | 3 |
| 343 | 46 | Anaheim | Lonnie Loach | `0x00546E` | 28 | 6 | 2 | 3 | 3 | 2 | 2 | 2 | L | 2 | 3 | 2 | 2 | 2 | 2 |
| 344 | 46 | Los Angeles | Lonnie Loach | `0x001FAE` | 29 | 6 | 2 | 3 | 3 | 2 | 2 | 2 | L | 2 | 3 | 2 | 2 | 2 | 2 |
| 345 | 46 | Edmonton | Martin Gelinas | `0x0019EA` | 7 | 8 | 3 | 3 | 2 | 2 | 3 | 2 | L | 3 | 2 | 2 | 4 | 2 | 2 |
| 346 | 46 | San Jose | Perry Berezan | `0x003D72` | 16 | 7 | 4 | 4 | 2 | 2 | 2 | 3 | R | 2 | 2 | 2 | 5 | 3 | 3 |
| 347 | 46 | Tampa Bay | Rob DiMaio | `0x00434C` | 18 | 5 | 2 | 3 | 3 | 3 | 2 | 2 | R | 2 | 2 | 3 | 2 | 2 | 3 |
| 348 | 46 | Toronto | Rob Pearson | `0x0046EE` | 12 | 6 | 2 | 2 | 3 | 3 | 1 | 1 | R | 2 | 3 | 4 | 5 | 2 | 4 |
| 349 | 46 | Detroit | Shawn Burr | `0x001682` | 11 | 6 | 2 | 2 | 3 | 3 | 2 | 4 | L | 2 | 2 | 3 | 1 | 2 | 3 |
| 350 | 46 | Edmonton | Shjon Podein | `0x00195E` | 26 | 9 | 2 | 2 | 3 | 2 | 2 | 2 | L | 2 | 4 | 2 | 5 | 1 | 2 |
| 351 | 45 | Vancouver | Adrien Plavsic | `0x004A54` | 6 | 7 | 2 | 2 | 3 | 3 | 2 | 3 | L | 2 | 2 | 3 | 0 | 2 | 3 |
| 352 | 45 | Calgary | Chris Dahlquist | `0x0011BC` | 5 | 8 | 3 | 3 | 1 | 4 | 3 | 3 | L | 3 | 1 | 3 | 5 | 2 | 3 |
| 353 | 45 | Edmonton | Chris Joseph | `0x001AE8` | 2 | 10 | 3 | 3 | 3 | 1 | 3 | 3 | R | 3 | 1 | 1 | 2 | 3 | 3 |
| 354 | 45 | Boston | David Shaw | `0x000BB4` | 34 | 9 | 2 | 2 | 2 | 4 | 2 | 3 | R | 2 | 2 | 4 | 5 | 2 | 3 |
| 355 | 45 | Toronto | Drake Berehowsky | `0x004776` | 55 | 10 | 2 | 2 | 3 | 3 | 2 | 3 | R | 2 | 2 | 3 | 0 | 2 | 3 |
| 356 | 45 | Calgary | Frank Musil | `0x00117C` | 3 | 9 | 4 | 4 | 2 | 3 | 2 | 3 | L | 2 | 1 | 3 | 5 | 2 | 3 |
| 357 | 45 | Vancouver | Garry Valk | `0x0049AA` | 23 | 7 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 3 | 3 | 3 | 2 | 3 |
| 358 | 45 | San Jose | Jay More | `0x003ECA` | 4 | 7 | 3 | 2 | 1 | 3 | 4 | 3 | R | 3 | 1 | 3 | 5 | 3 | 4 |
| 359 | 45 | Vancouver | Jim Sandlak | `0x0049FE` | 25 | 11 | 1 | 2 | 3 | 3 | 3 | 2 | R | 2 | 2 | 3 | 3 | 2 | 4 |
| 360 | 45 | Ottawa | Jody Hull | `0x00323C` | 17 | 9 | 2 | 2 | 3 | 3 | 2 | 3 | R | 2 | 2 | 3 | 3 | 2 | 1 |
| 361 | 45 | Florida | Joe Cirella | `0x00527A` | 6 | 10 | 3 | 3 | 2 | 2 | 3 | 2 | R | 2 | 2 | 3 | 3 | 2 | 3 |
| 362 | 45 | New York Rangers | Joe Cirella | `0x00300A` | 6 | 10 | 3 | 3 | 2 | 2 | 3 | 2 | R | 2 | 2 | 3 | 3 | 2 | 3 |
| 363 | 45 | Winnipeg | John Druce | `0x004CDE` | 15 | 9 | 2 | 2 | 3 | 3 | 2 | 3 | R | 2 | 2 | 3 | 1 | 2 | 2 |
| 364 | 45 | Pittsburgh | Martin Straka | `0x0037FC` | 82 | 5 | 3 | 3 | 3 | 2 | 2 | 2 | L | 2 | 2 | 2 | 0 | 3 | 2 |
| 365 | 45 | Ottawa | Neil Brady | `0x003146` | 12 | 9 | 2 | 2 | 3 | 3 | 2 | 2 | L | 2 | 2 | 3 | 1 | 3 | 3 |
| 366 | 45 | Hartford | Randy Cunnyworth | `0x001CF4` | 7 | 6 | 2 | 3 | 2 | 3 | 2 | 3 | L | 2 | 2 | 3 | 5 | 3 | 3 |
| 367 | 45 | St. Louis | Rich Sutter | `0x0040F4` | 23 | 7 | 3 | 3 | 2 | 4 | 1 | 2 | R | 2 | 2 | 4 | 5 | 2 | 3 |
| 368 | 45 | St. Louis | Ron Wilson | `0x004030` | 18 | 6 | 3 | 3 | 2 | 4 | 1 | 2 | L | 2 | 2 | 4 | 3 | 2 | 2 |
| 369 | 45 | Anaheim | Steven King | `0x0053EC` | 27 | 7 | 2 | 2 | 3 | 2 | 2 | 1 | R | 2 | 3 | 3 | 4 | 2 | 2 |
| 370 | 45 | New York Rangers | Steven King | `0x002ED2` | 25 | 7 | 2 | 2 | 3 | 2 | 2 | 1 | R | 2 | 3 | 3 | 4 | 2 | 2 |
| 371 | 44 | Washington | Alan May | `0x004F7C` | 16 | 9 | 2 | 2 | 2 | 3 | 3 | 3 | R | 2 | 2 | 3 | 4 | 2 | 5 |
| 372 | 44 | Montreal | Benoit Brunet | `0x0025A4` | 22 | 6 | 2 | 2 | 3 | 2 | 2 | 2 | L | 2 | 3 | 2 | 2 | 2 | 2 |
| 373 | 44 | Anaheim | Bill Houlder | `0x00550E` | 33 | 11 | 3 | 2 | 3 | 3 | 2 | 3 | L | 2 | 2 | 2 | 4 | 2 | 1 |
| 374 | 44 | Boston | Brent Hughes | `0x000AD8` | 42 | 6 | 3 | 3 | 1 | 3 | 2 | 3 | L | 3 | 2 | 2 | 5 | 3 | 4 |
| 375 | 44 | Boston | Gordie Roberts | `0x000BDE` | 14 | 7 | 3 | 2 | 2 | 3 | 1 | 3 | L | 2 | 3 | 3 | 0 | 2 | 3 |
| 376 | 44 | San Jose | Jeff Odgers | `0x003DB8` | 36 | 9 | 1 | 2 | 3 | 2 | 2 | 2 | L | 2 | 3 | 3 | 4 | 1 | 5 |
| 377 | 44 | Toronto | Mark Osborne | `0x004690` | 21 | 9 | 2 | 2 | 2 | 4 | 2 | 3 | L | 2 | 2 | 4 | 4 | 1 | 3 |
| 378 | 44 | San Jose | Mark Pederson | `0x003E26` | 18 | 8 | 2 | 2 | 3 | 2 | 2 | 2 | L | 2 | 3 | 2 | 5 | 2 | 2 |
| 379 | 44 | Hartford | Nick Kypreos | `0x001D3C` | 20 | 8 | 2 | 2 | 2 | 2 | 1 | 3 | L | 2 | 4 | 3 | 4 | 1 | 5 |
| 380 | 44 | New York Rangers | Peter Andersson | `0x002FDC` | 5 | 9 | 2 | 3 | 3 | 2 | 2 | 2 | L | 3 | 1 | 2 | 4 | 3 | 2 |
| 381 | 44 | Pittsburgh | Peter Taglianeti | `0x003896` | 32 | 9 | 2 | 2 | 2 | 3 | 3 | 4 | L | 3 | 0 | 4 | 3 | 2 | 4 |
| 382 | 44 | Calgary | Roger Johansson | `0x00114C` | 34 | 7 | 4 | 3 | 2 | 2 | 2 | 3 | L | 3 | 1 | 2 | 3 | 3 | 2 |
| 383 | 44 | Tampa Bay | Roman Hamrlik | `0x00445A` | 44 | 7 | 3 | 2 | 2 | 3 | 3 | 2 | L | 3 | 1 | 3 | 4 | 2 | 3 |
| 384 | 44 | Chicago | Stephane Matteau | `0x001364` | 32 | 8 | 2 | 2 | 3 | 2 | 1 | 3 | L | 2 | 3 | 3 | 2 | 1 | 3 |
| 385 | 44 | Detroit | Steve Konroyd | `0x0017E2` | 8 | 8 | 3 | 3 | 2 | 3 | 2 | 4 | L | 2 | 1 | 3 | 2 | 3 | 3 |
| 386 | 44 | Quebec | Steven Finn | `0x003BC8` | 29 | 8 | 2 | 2 | 2 | 3 | 3 | 3 | L | 2 | 2 | 3 | 4 | 2 | 4 |
| 387 | 43 | Ottawa | Andrew McBain | `0x003250` | 20 | 9 | 2 | 2 | 3 | 3 | 2 | 3 | R | 2 | 2 | 2 | 1 | 2 | 2 |
| 388 | 43 | Toronto | Bob Rouse | `0x0047A6` | 3 | 10 | 3 | 3 | 2 | 4 | 2 | 4 | R | 2 | 0 | 4 | 4 | 2 | 3 |
| 389 | 43 | New York Islanders | Brad Dalgarno | `0x002C14` | 15 | 11 | 1 | 2 | 3 | 2 | 1 | 1 | R | 2 | 4 | 2 | 1 | 2 | 3 |
| 390 | 43 | Chicago | Bryan Marchment | `0x001474` | 2 | 8 | 3 | 3 | 2 | 4 | 2 | 4 | L | 2 | 1 | 3 | 2 | 1 | 5 |
| 391 | 43 | Pittsburgh | Dave Tippett | `0x00377C` | 14 | 6 | 2 | 3 | 2 | 3 | 2 | 2 | L | 2 | 2 | 3 | 1 | 2 | 2 |
| 392 | 43 | Edmonton | Geoff Smith | `0x001AA2` | 25 | 9 | 4 | 3 | 2 | 3 | 1 | 1 | L | 3 | 1 | 3 | 2 | 3 | 1 |
| 393 | 43 | Chicago | Keith Brown | `0x0014BC` | 4 | 7 | 2 | 2 | 2 | 3 | 2 | 3 | R | 3 | 1 | 3 | 4 | 3 | 3 |
| 394 | 43 | Washington | Keith Jones | `0x004FD0` | 26 | 7 | 2 | 2 | 2 | 2 | 2 | 3 | R | 2 | 3 | 2 | 2 | 3 | 4 |
| 395 | 43 | Pittsburgh | Mike Ramsey | `0x0038B0` | 6 | 8 | 3 | 2 | 2 | 3 | 2 | 3 | L | 2 | 2 | 3 | 1 | 2 | 2 |
| 396 | 43 | San Jose | Neil Wilkinson | `0x003E54` | 5 | 7 | 3 | 3 | 1 | 3 | 2 | 3 | R | 3 | 0 | 4 | 4 | 4 | 3 |
| 397 | 43 | New Jersey | Tom Chorske | `0x0028AA` | 9 | 9 | 2 | 2 | 3 | 2 | 1 | 2 | R | 3 | 2 | 2 | 2 | 3 | 2 |
| 398 | 42 | Calgary | Alexnder Godynyuk | `0x0011D6` | 21 | 10 | 2 | 3 | 2 | 2 | 2 | 3 | L | 2 | 2 | 3 | 5 | 1 | 2 |
| 399 | 42 | Florida | Alexnder Godynyuk | `0x005290` | 21 | 10 | 2 | 3 | 2 | 2 | 2 | 3 | L | 2 | 2 | 3 | 5 | 1 | 2 |
| 400 | 42 | Chicago | Cam Russell | `0x0014D2` | 8 | 5 | 3 | 3 | 1 | 3 | 2 | 3 | L | 3 | 1 | 3 | 5 | 2 | 4 |
| 401 | 42 | Buffalo | Dave Hannan | `0x000D64` | 14 | 6 | 2 | 2 | 2 | 3 | 2 | 2 | L | 2 | 3 | 2 | 0 | 2 | 2 |
| 402 | 42 | Florida | Dave Lowry | `0x00520A` | 10 | 8 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 2 | 3 | 4 | 2 | 4 |
| 403 | 42 | St. Louis | Dave Lowry | `0x004086` | 10 | 8 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 2 | 3 | 4 | 2 | 4 |
| 404 | 42 | Philadelphia | Dave Snuggerud | `0x0034F8` | 14 | 7 | 3 | 3 | 2 | 2 | 2 | 2 | L | 3 | 1 | 2 | 5 | 3 | 1 |
| 405 | 42 | San Jose | Dean Evason | `0x003D44` | 12 | 6 | 1 | 1 | 2 | 3 | 2 | 3 | R | 2 | 3 | 3 | 3 | 2 | 3 |
| 406 | 42 | Quebec | Gino Cavallini | `0x003A86` | 44 | 11 | 1 | 3 | 2 | 2 | 2 | 2 | L | 2 | 3 | 2 | 2 | 2 | 2 |
| 407 | 42 | Winnipeg | Igor Ulanov | `0x004D7C` | 5 | 9 | 2 | 2 | 2 | 4 | 2 | 3 | L | 2 | 1 | 4 | 0 | 2 | 4 |
| 408 | 42 | Hartford | Jamie Leach | `0x001D66` | 34 | 8 | 3 | 3 | 2 | 1 | 2 | 3 | R | 2 | 3 | 1 | 5 | 2 | 0 |
| 409 | 42 | New Jersey | Janne Ojanen | `0x00284E` | 34 | 9 | 2 | 2 | 3 | 2 | 2 | 2 | L | 2 | 2 | 2 | 2 | 3 | 2 |
| 410 | 42 | Edmonton | Kelly Buchberger | `0x0019D0` | 16 | 10 | 2 | 2 | 2 | 3 | 2 | 4 | L | 1 | 3 | 3 | 2 | 1 | 3 |
| 411 | 42 | Edmonton | Mike Hudson | `0x001974` | 20 | 9 | 3 | 3 | 2 | 4 | 2 | 4 | L | 2 | 0 | 3 | 2 | 3 | 3 |
| 412 | 42 | Ottawa | Mike Peluso | `0x0031D0` | 44 | 9 | 2 | 2 | 2 | 3 | 2 | 3 | L | 1 | 3 | 3 | 5 | 2 | 5 |
| 413 | 42 | Los Angeles | Pat Conacher | `0x001FC4` | 15 | 7 | 2 | 2 | 2 | 2 | 2 | 2 | L | 2 | 3 | 2 | 4 | 3 | 1 |
| 414 | 42 | Vancouver | Robert Dirk | `0x004AC4` | 22 | 11 | 2 | 2 | 1 | 3 | 2 | 3 | L | 3 | 2 | 3 | 3 | 2 | 4 |
| 415 | 42 | Washington | Todd Krygier | `0x004F66` | 21 | 6 | 2 | 3 | 2 | 3 | 2 | 2 | L | 2 | 2 | 3 | 5 | 1 | 2 |
| 416 | 42 | Calgary | Trent Yawney | `0x001166` | 18 | 7 | 3 | 2 | 2 | 4 | 2 | 3 | L | 3 | 0 | 3 | 1 | 3 | 3 |
| 417 | 41 | Buffalo | Brad May | `0x000DA4` | 27 | 9 | 2 | 2 | 2 | 2 | 2 | 3 | L | 2 | 3 | 2 | 5 | 1 | 4 |
| 418 | 41 | New York Islanders | Claude Loiselle | `0x002B8C` | 10 | 8 | 3 | 2 | 2 | 1 | 1 | 3 | L | 3 | 3 | 1 | 5 | 2 | 4 |
| 419 | 41 | New York Islanders | Dan Marois | `0x002C44` | 17 | 7 | 3 | 3 | 2 | 1 | 3 | 1 | R | 3 | 1 | 2 | 4 | 2 | 3 |
| 420 | 41 | Dallas | Derian Hatcher | `0x00238A` | 2 | 9 | 3 | 2 | 2 | 3 | 2 | 2 | L | 2 | 1 | 4 | 2 | 2 | 4 |
| 421 | 41 | Philadelphia | Doug Evans | `0x0034CC` | 15 | 6 | 2 | 2 | 2 | 2 | 2 | 2 | L | 2 | 3 | 2 | 2 | 2 | 3 |
| 422 | 41 | Pittsburgh | Mike Needham | `0x003814` | 39 | 6 | 2 | 2 | 2 | 2 | 2 | 2 | R | 2 | 3 | 2 | 5 | 2 | 1 |
| 423 | 41 | San Jose | Mike Sullivan | `0x003D5A` | 47 | 6 | 2 | 3 | 2 | 3 | 2 | 1 | L | 2 | 1 | 4 | 5 | 2 | 1 |
| 424 | 41 | Tampa Bay | Shawn Chambers | `0x004472` | 22 | 9 | 2 | 2 | 3 | 3 | 2 | 2 | L | 2 | 1 | 3 | 2 | 2 | 2 |
| 425 | 41 | Florida | Tom Fitzgerald | `0x0051AC` | 14 | 8 | 2 | 2 | 2 | 3 | 2 | 2 | R | 2 | 2 | 3 | 2 | 2 | 2 |
| 426 | 41 | New York Islanders | Tom Fitzgerald | `0x002C2C` | 14 | 8 | 2 | 2 | 2 | 3 | 2 | 2 | R | 2 | 2 | 3 | 2 | 2 | 2 |
| 427 | 41 | San Jose | Tom Pederson | `0x003E86` | 41 | 4 | 1 | 2 | 3 | 2 | 2 | 2 | R | 2 | 2 | 2 | 4 | 3 | 2 |
| 428 | 41 | Anaheim | Troy Loney | `0x005402` | 24 | 10 | 2 | 3 | 2 | 3 | 2 | 4 | L | 2 | 1 | 3 | 2 | 1 | 3 |
| 429 | 41 | Pittsburgh | Troy Loney | `0x003792` | 24 | 10 | 2 | 3 | 2 | 3 | 2 | 4 | L | 2 | 1 | 3 | 2 | 1 | 3 |
| 430 | 40 | Ottawa | David Archibald | `0x003188` | 15 | 7 | 2 | 2 | 2 | 3 | 2 | 1 | L | 2 | 2 | 3 | 5 | 2 | 2 |
| 431 | 40 | Chicago | Frantsek Kucera | `0x00148E` | 6 | 9 | 2 | 2 | 2 | 4 | 3 | 1 | R | 2 | 1 | 3 | 4 | 2 | 2 |
| 432 | 40 | Tampa Bay | Joe Reekie | `0x0044A2` | 29 | 11 | 3 | 2 | 2 | 2 | 3 | 2 | L | 3 | 0 | 3 | 2 | 2 | 3 |
| 433 | 40 | St. Louis | Lee Norwood | `0x004192` | 20 | 8 | 2 | 2 | 2 | 3 | 1 | 3 | L | 2 | 2 | 3 | 2 | 2 | 4 |
| 434 | 40 | New Jersey | Randy McKay | `0x002920` | 21 | 6 | 2 | 2 | 2 | 2 | 2 | 2 | R | 2 | 2 | 3 | 4 | 2 | 4 |
| 435 | 40 | Dallas | Richard Matvichuk | `0x0023B8` | 4 | 7 | 2 | 3 | 1 | 3 | 3 | 3 | L | 3 | 0 | 3 | 5 | 2 | 2 |
| 436 | 40 | Winnipeg | Russ Romaniuk | `0x004C7E` | 21 | 6 | 3 | 2 | 1 | 2 | 2 | 3 | L | 2 | 3 | 2 | 5 | 2 | 2 |
| 437 | 40 | St. Louis | Stephane Quintal | `0x004178` | 33 | 11 | 2 | 2 | 1 | 4 | 2 | 3 | R | 3 | 0 | 4 | 4 | 3 | 3 |
| 438 | 39 | Ottawa | Doug Smail | `0x0031E6` | 9 | 5 | 3 | 3 | 2 | 3 | 2 | 2 | L | 2 | 1 | 2 | 4 | 2 | 3 |
| 439 | 39 | San Jose | Doug Zmolek | `0x003E9C` | 19 | 12 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 1 | 3 | 5 | 2 | 4 |
| 440 | 39 | San Jose | Ed Courtenay | `0x003E10` | 39 | 9 | 2 | 2 | 3 | 3 | 1 | 1 | R | 1 | 3 | 2 | 1 | 2 | 1 |
| 441 | 39 | Vancouver | Gino Odjick | `0x004994` | 29 | 11 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 1 | 3 | 3 | 2 | 6 |
| 442 | 39 | Pittsburgh | Jim Paek | `0x00386E` | 2 | 8 | 2 | 3 | 2 | 2 | 2 | 2 | L | 2 | 1 | 3 | 1 | 2 | 2 |
| 443 | 39 | Buffalo | Ken Sutton | `0x000E9A` | 41 | 8 | 2 | 2 | 2 | 4 | 2 | 1 | L | 1 | 2 | 3 | 2 | 3 | 2 |
| 444 | 39 | Edmonton | Luke Richardson | `0x001ACE` | 22 | 11 | 3 | 3 | 1 | 3 | 2 | 4 | L | 3 | 0 | 2 | 4 | 3 | 4 |
| 445 | 39 | St. Louis | Rick Zombo | `0x004164` | 4 | 8 | 2 | 2 | 2 | 3 | 3 | 2 | R | 2 | 0 | 4 | 0 | 2 | 3 |
| 446 | 39 | Calgary | Ronnie Stern | `0x00110C` | 22 | 8 | 2 | 2 | 2 | 1 | 2 | 3 | R | 2 | 3 | 1 | 3 | 2 | 4 |
| 447 | 39 | Washington | Steve Konowlchuk | `0x004F1C` | 22 | 6 | 2 | 2 | 2 | 2 | 2 | 3 | L | 2 | 2 | 2 | 2 | 2 | 2 |
| 448 | 39 | Tampa Bay | Steve Maltais | `0x0043D6` | 37 | 10 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 1 | 3 | 4 | 2 | 2 |
| 449 | 39 | Philadelphia | Terry Carkner | `0x0035B0` | 29 | 10 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 1 | 3 | 0 | 2 | 4 |
| 450 | 39 | Vancouver | Tim Hunter | `0x004A14` | 26 | 9 | 2 | 2 | 1 | 2 | 2 | 3 | R | 2 | 3 | 2 | 3 | 2 | 4 |
| 451 | 39 | Ottawa | Tomas Jelinek | `0x003268` | 25 | 7 | 2 | 2 | 2 | 2 | 2 | 1 | L | 3 | 2 | 2 | 5 | 1 | 3 |
| 452 | 39 | New Jersey | Tommy Albelin | `0x0029F4` | 6 | 7 | 3 | 2 | 2 | 3 | 2 | 2 | L | 3 | 0 | 3 | 3 | 2 | 2 |
| 453 | 39 | New York Islanders | Travis Green | `0x002B60` | 39 | 8 | 2 | 2 | 3 | 2 | 2 | 3 | R | 2 | 1 | 2 | 3 | 2 | 2 |
| 454 | 38 | Quebec | Adam Foote | `0x003BB4` | 52 | 6 | 2 | 2 | 2 | 4 | 1 | 3 | R | 2 | 1 | 3 | 2 | 2 | 4 |
| 455 | 38 | Anaheim | Bob Corkum | `0x005440` | 30 | 10 | 2 | 2 | 1 | 3 | 2 | 2 | R | 2 | 2 | 3 | 5 | 2 | 2 |
| 456 | 38 | Buffalo | Bob Corkum | `0x000E26` | 29 | 10 | 2 | 2 | 1 | 3 | 2 | 2 | R | 2 | 2 | 3 | 5 | 2 | 2 |
| 457 | 38 | Toronto | Bob McGill | `0x0047D4` | 8 | 8 | 2 | 2 | 0 | 2 | 3 | 3 | R | 2 | 3 | 2 | 3 | 2 | 4 |
| 458 | 38 | St. Louis | Igor Korolev | `0x004070` | 38 | 7 | 2 | 2 | 2 | 3 | 2 | 2 | L | 2 | 1 | 3 | 0 | 2 | 1 |
| 459 | 38 | Pittsburgh | Jeff Daniels | `0x0037A6` | 20 | 9 | 2 | 2 | 1 | 2 | 2 | 2 | L | 2 | 3 | 2 | 4 | 2 | 1 |
| 460 | 38 | Buffalo | Keith Carney | `0x000EF4` | 6 | 9 | 2 | 3 | 2 | 1 | 2 | 3 | L | 2 | 2 | 1 | 4 | 2 | 4 |
| 461 | 38 | Montreal | Kevin Haller | `0x00268A` | 14 | 6 | 2 | 2 | 2 | 3 | 1 | 2 | L | 1 | 2 | 4 | 5 | 2 | 3 |
| 462 | 38 | Montreal | Lyle Odelein | `0x0026CE` | 24 | 9 | 3 | 2 | 2 | 4 | 2 | 3 | L | 2 | 0 | 3 | 3 | 2 | 4 |
| 463 | 38 | Dallas | Stewart Gavin | `0x002330` | 12 | 7 | 3 | 3 | 2 | 2 | 1 | 2 | L | 2 | 2 | 2 | 5 | 1 | 3 |
| 464 | 38 | New Jersey | Troy Mallette | `0x0028C0` | 8 | 10 | 2 | 2 | 2 | 2 | 1 | 1 | L | 2 | 4 | 1 | 3 | 1 | 3 |
| 465 | 38 | Hartford | Yvon Corriveau | `0x001CC6` | 11 | 8 | 2 | 2 | 2 | 3 | 2 | 3 | L | 1 | 2 | 3 | 3 | 1 | 1 |
| 466 | 37 | Florida | Bill Lindsay | `0x005264` | 22 | 7 | 1 | 2 | 2 | 2 | 2 | 2 | L | 2 | 2 | 2 | 4 | 2 | 1 |
| 467 | 37 | Detroit | Brad McCrimmon | `0x0017FA` | 2 | 8 | 3 | 2 | 2 | 4 | 2 | 4 | L | 1 | 0 | 3 | 1 | 3 | 3 |
| 468 | 37 | Florida | Gord Hynes | `0x0052C2` | 26 | 4 | 2 | 2 | 2 | 2 | 1 | 3 | L | 2 | 2 | 2 | 5 | 2 | 2 |
| 469 | 37 | Philadelphia | Gord Hynes | `0x0035F4` | 26 | 4 | 2 | 2 | 2 | 2 | 1 | 3 | L | 2 | 2 | 2 | 5 | 2 | 2 |
| 470 | 37 | New York Rangers | Jeff Beukeboom | `0x002FB0` | 23 | 11 | 2 | 2 | 2 | 4 | 2 | 3 | R | 2 | 0 | 3 | 0 | 2 | 4 |
| 471 | 37 | Florida | Jesse Belanger | `0x00524C` | 29 | 4 | 1 | 2 | 3 | 1 | 2 | 2 | L | 2 | 2 | 1 | 4 | 2 | 1 |
| 472 | 37 | Tampa Bay | Marc Bergevin | `0x00448A` | 25 | 6 | 2 | 2 | 2 | 2 | 3 | 3 | L | 2 | 0 | 3 | 3 | 2 | 2 |
| 473 | 37 | Pittsburgh | Mike Stapleton | `0x00374C` | 26 | 6 | 3 | 2 | 2 | 2 | 2 | 2 | R | 2 | 1 | 2 | 5 | 3 | 0 |
| 474 | 37 | New York Islanders | Richard Pilon | `0x002D14` | 47 | 9 | 3 | 3 | 1 | 2 | 3 | 3 | L | 2 | 1 | 2 | 5 | 1 | 5 |
| 475 | 37 | Philadelphia | Ryan McGill | `0x0035DE` | 27 | 8 | 2 | 2 | 2 | 3 | 2 | 2 | R | 2 | 1 | 3 | 4 | 1 | 4 |
| 476 | 37 | Anaheim | Tim Sweeney | `0x005498` | 41 | 5 | 2 | 2 | 3 | 2 | 2 | 1 | L | 2 | 2 | 1 | 4 | 1 | 1 |
| 477 | 36 | Hartford | Allen Pedersen | `0x001E06` | 41 | 10 | 2 | 3 | 1 | 3 | 2 | 3 | L | 2 | 1 | 2 | 1 | 2 | 3 |
| 478 | 36 | St. Louis | Basil McRae | `0x00409A` | 17 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | L | 2 | 1 | 3 | 4 | 1 | 5 |
| 479 | 36 | Chicago | Craig Muni | `0x0014A8` | 3 | 9 | 2 | 2 | 1 | 3 | 2 | 4 | L | 2 | 0 | 4 | 2 | 2 | 3 |
| 480 | 36 | Ottawa | Darren Rumble | `0x0032AA` | 34 | 9 | 2 | 2 | 2 | 3 | 2 | 3 | L | 2 | 0 | 3 | 4 | 2 | 3 |
| 481 | 36 | Montreal | Ed Ronan | `0x00262A` | 31 | 8 | 2 | 2 | 2 | 1 | 2 | 3 | R | 2 | 2 | 1 | 4 | 2 | 1 |
| 482 | 36 | Detroit | Jim Hiller | `0x00173E` | 14 | 7 | 2 | 2 | 2 | 1 | 2 | 3 | R | 2 | 2 | 1 | 4 | 2 | 4 |
| 483 | 36 | Ottawa | Ken Hammond | `0x0032D8` | 5 | 7 | 3 | 2 | 1 | 3 | 2 | 3 | L | 2 | 1 | 3 | 5 | 1 | 3 |
| 484 | 36 | San Jose | Peter Ahola | `0x003EDC` | 21 | 9 | 2 | 2 | 1 | 3 | 2 | 3 | L | 2 | 1 | 3 | 4 | 2 | 2 |
| 485 | 36 | Florida | Randy Gilhen | `0x005236` | 20 | 7 | 2 | 2 | 1 | 3 | 2 | 3 | L | 2 | 1 | 3 | 5 | 2 | 1 |
| 486 | 36 | Tampa Bay | Randy Gilhen | `0x004390` | 20 | 7 | 2 | 2 | 1 | 3 | 2 | 3 | L | 2 | 1 | 3 | 5 | 2 | 1 |
| 487 | 36 | San Jose | Rob Zettler | `0x003EF2` | 2 | 7 | 2 | 2 | 1 | 3 | 2 | 2 | L | 3 | 0 | 3 | 4 | 3 | 4 |
| 488 | 35 | Hartford | Dan Keczmer | `0x001DC2` | 37 | 7 | 2 | 2 | 2 | 3 | 1 | 1 | L | 2 | 2 | 2 | 5 | 1 | 3 |
| 489 | 35 | Boston | Darin Kimble | `0x000B46` | 29 | 9 | 1 | 1 | 2 | 1 | 1 | 2 | R | 2 | 4 | 1 | 3 | 1 | 5 |
| 490 | 35 | Winnipeg | Dean Kennedy | `0x004DA6` | 26 | 9 | 2 | 2 | 1 | 3 | 2 | 3 | R | 2 | 0 | 4 | 4 | 2 | 3 |
| 491 | 35 | San Jose | John Carter | `0x003DCE` | 20 | 4 | 2 | 2 | 2 | 2 | 2 | 2 | L | 2 | 1 | 2 | 5 | 2 | 3 |
| 492 | 35 | Edmonton | Louie DeBrusk | `0x001A02` | 29 | 12 | 1 | 1 | 2 | 1 | 2 | 3 | L | 1 | 4 | 1 | 5 | 1 | 5 |
| 493 | 35 | Los Angeles | Mark Hardy | `0x0020C0` | 24 | 8 | 3 | 3 | 2 | 3 | 1 | 3 | L | 2 | 0 | 2 | 1 | 2 | 3 |
| 494 | 35 | Winnipeg | Mike Lalor | `0x004D92` | 22 | 9 | 3 | 2 | 1 | 3 | 2 | 2 | L | 2 | 0 | 4 | 5 | 2 | 3 |
| 495 | 35 | Anaheim | Randy Ladouceur | `0x0054DC` | 39 | 11 | 3 | 3 | 1 | 3 | 1 | 2 | L | 2 | 1 | 3 | 5 | 1 | 4 |
| 496 | 35 | Hartford | Randy Ladouceur | `0x001DEC` | 29 | 11 | 3 | 3 | 1 | 3 | 1 | 2 | L | 2 | 1 | 3 | 5 | 1 | 4 |
| 497 | 35 | Washington | Reggie Savage | `0x004F36` | 15 | 7 | 2 | 2 | 2 | 1 | 2 | 2 | L | 2 | 2 | 1 | 3 | 2 | 2 |
| 498 | 35 | Buffalo | Rob Ray | `0x000DCA` | 32 | 10 | 3 | 4 | 1 | 1 | 1 | 3 | L | 2 | 2 | 1 | 5 | 1 | 4 |
| 499 | 35 | Anaheim | Sean Hill | `0x0054C8` | 38 | 8 | 2 | 2 | 2 | 2 | 2 | 3 | R | 2 | 1 | 2 | 3 | 1 | 4 |
| 500 | 35 | Montreal | Sean Hill | `0x0026E4` | 38 | 8 | 2 | 2 | 2 | 2 | 2 | 3 | R | 2 | 1 | 2 | 3 | 1 | 4 |
| 501 | 35 | Dallas | Shane Churla | `0x00231A` | 27 | 9 | 1 | 2 | 2 | 2 | 1 | 2 | R | 2 | 2 | 2 | 1 | 2 | 5 |
| 502 | 35 | Tampa Bay | Tim Bergland | `0x00441A` | 21 | 8 | 2 | 2 | 1 | 3 | 2 | 2 | R | 2 | 1 | 3 | 5 | 2 | 2 |
| 503 | 34 | Quebec | Bill Lindsay | `0x003AB6` | 20 | 6 | 1 | 2 | 2 | 2 | 2 | 2 | L | 2 | 1 | 2 | 3 | 2 | 1 |
| 504 | 34 | Philadelphia | Claude Boivin | `0x003510` | 10 | 9 | 1 | 1 | 2 | 1 | 2 | 2 | L | 1 | 4 | 1 | 2 | 1 | 4 |
| 505 | 34 | New Jersey | Ken Daneyko | `0x0029DE` | 3 | 10 | 2 | 2 | 1 | 4 | 1 | 4 | L | 2 | 0 | 4 | 4 | 1 | 4 |
| 506 | 34 | St. Louis | Murray Baron | `0x0041D2` | 34 | 10 | 3 | 3 | 1 | 2 | 2 | 2 | L | 2 | 1 | 2 | 5 | 1 | 3 |
| 507 | 34 | Buffalo | Randy Moller | `0x000EDE` | 24 | 10 | 3 | 2 | 2 | 1 | 2 | 3 | R | 2 | 1 | 1 | 1 | 2 | 4 |
| 508 | 34 | Ottawa | Rob Murphy | `0x0031A2` | 18 | 9 | 1 | 2 | 2 | 2 | 2 | 3 | L | 2 | 1 | 2 | 4 | 1 | 2 |
| 509 | 34 | Anaheim | Robin Bawa | `0x005484` | 26 | 11 | 2 | 2 | 1 | 1 | 2 | 1 | R | 1 | 4 | 1 | 3 | 2 | 3 |
| 510 | 34 | San Jose | Robin Bawa | `0x003D8A` | 26 | 11 | 2 | 2 | 1 | 1 | 2 | 1 | R | 1 | 4 | 1 | 3 | 2 | 3 |
| 511 | 34 | Washington | Shawn Anderson | `0x005072` | 36 | 9 | 2 | 2 | 1 | 2 | 2 | 3 | L | 2 | 1 | 2 | 3 | 3 | 1 |
| 512 | 34 | Toronto | Sylvain Lefebvre | `0x0047BA` | 2 | 9 | 2 | 2 | 1 | 3 | 1 | 3 | L | 2 | 0 | 4 | 4 | 3 | 3 |
| 513 | 34 | Dallas | Trent Klatt | `0x002304` | 29 | 9 | 2 | 2 | 3 | 1 | 2 | 1 | R | 2 | 1 | 1 | 1 | 2 | 2 |
| 514 | 33 | Dallas | Craig Ludwig | `0x0023A2` | 3 | 12 | 2 | 2 | 1 | 3 | 2 | 3 | L | 2 | 0 | 3 | 4 | 2 | 4 |
| 515 | 33 | Buffalo | Grant Ledyard | `0x000EAE` | 3 | 9 | 4 | 3 | 2 | 3 | 1 | 3 | L | 1 | 0 | 2 | 3 | 2 | 3 |
| 516 | 33 | New York Rangers | Jay Wells | `0x002FF6` | 24 | 10 | 2 | 2 | 2 | 2 | 2 | 3 | L | 2 | 0 | 2 | 1 | 2 | 4 |
| 517 | 33 | Hartford | Jim McKenzie | `0x001CDE` | 33 | 9 | 2 | 2 | 1 | 1 | 2 | 2 | L | 2 | 2 | 2 | 3 | 1 | 4 |
| 518 | 33 | Florida | Milan Tichy | `0x0052D6` | 43 | 8 | 3 | 2 | 1 | 1 | 2 | 3 | L | 2 | 1 | 2 | 2 | 2 | 4 |
| 519 | 33 | St. Louis | Philippe Bozon | `0x004058` | 36 | 6 | 2 | 2 | 2 | 2 | 2 | 1 | L | 2 | 1 | 2 | 5 | 1 | 3 |
| 520 | 33 | Winnipeg | Randy Carlyle | `0x004DBC` | 8 | 9 | 2 | 2 | 1 | 2 | 2 | 3 | L | 2 | 1 | 2 | 5 | 2 | 2 |
| 521 | 32 | Winnipeg | Andy Brickley | `0x004C96` | 23 | 9 | 2 | 2 | 2 | 2 | 1 | 3 | L | 2 | 0 | 2 | 0 | 3 | 0 |
| 522 | 32 | Montreal | Donald Dufresne | `0x0026F8` | 34 | 9 | 2 | 2 | 1 | 2 | 2 | 3 | R | 2 | 1 | 2 | 3 | 1 | 3 |
| 523 | 32 | New York Rangers | Joey Kocur | `0x002F42` | 26 | 8 | 2 | 1 | 1 | 3 | 2 | 3 | R | 2 | 1 | 2 | 4 | 2 | 4 |
| 524 | 32 | Philadelphia | Vachslav Butsayev | `0x00349A` | 22 | 6 | 2 | 2 | 2 | 2 | 2 | 2 | L | 2 | 0 | 2 | 1 | 2 | 3 |
| 525 | 31 | Ottawa | Darcy Loewen | `0x003210` | 10 | 6 | 2 | 2 | 1 | 2 | 2 | 1 | L | 1 | 2 | 2 | 4 | 2 | 4 |
| 526 | 31 | Anaheim | David Williams | `0x0054F6` | 3 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | R | 2 | 0 | 2 | 2 | 1 | 3 |
| 527 | 31 | San Jose | David Williams | `0x003EB2` | 3 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | R | 2 | 0 | 2 | 2 | 1 | 3 |
| 528 | 31 | Hartford | Doug Houda | `0x001DD8` | 27 | 7 | 2 | 2 | 1 | 2 | 2 | 2 | R | 2 | 1 | 2 | 4 | 1 | 4 |
| 529 | 31 | Buffalo | Gord Donnelly | `0x000EC6` | 34 | 9 | 1 | 2 | 2 | 3 | 1 | 3 | R | 1 | 1 | 2 | 2 | 2 | 5 |
| 530 | 31 | Washington | Rod Langway | `0x0050A2` | 5 | 11 | 3 | 2 | 0 | 2 | 2 | 3 | L | 3 | 0 | 2 | 3 | 2 | 3 |
| 531 | 31 | Edmonton | Steven Rice | `0x001A46` | 12 | 11 | 2 | 2 | 2 | 1 | 2 | 1 | R | 2 | 1 | 1 | 2 | 2 | 3 |
| 532 | 30 | St. Louis | Curt Giles | `0x0041BE` | 2 | 5 | 3 | 2 | 1 | 3 | 1 | 2 | L | 2 | 0 | 3 | 3 | 1 | 2 |
| 533 | 30 | Los Angeles | Gary Shuchuk | `0x001F6A` | 14 | 6 | 2 | 2 | 2 | 0 | 1 | 2 | R | 2 | 2 | 0 | 3 | 2 | 2 |
| 534 | 30 | Pittsburgh | Grant Jennings | `0x0038E0` | 3 | 9 | 2 | 2 | 1 | 3 | 2 | 3 | L | 2 | 0 | 2 | 3 | 1 | 3 |
| 535 | 30 | New York Rangers | Paul Broten | `0x002F2C` | 37 | 7 | 1 | 1 | 2 | 3 | 1 | 2 | R | 1 | 2 | 2 | 3 | 1 | 2 |
| 536 | 30 | Los Angeles | Warren Rychel | `0x001FDA` | 10 | 7 | 2 | 2 | 2 | 1 | 1 | 3 | L | 1 | 2 | 1 | 5 | 1 | 5 |
| 537 | 29 | Dallas | Brad Berry | `0x0023EA` | 5 | 7 | 2 | 2 | 0 | 3 | 2 | 3 | L | 2 | 0 | 3 | 5 | 1 | 4 |
| 538 | 29 | Quebec | Craig Wolanin | `0x003BFA` | 6 | 9 | 2 | 2 | 2 | 2 | 1 | 2 | L | 1 | 1 | 2 | 1 | 1 | 4 |
| 539 | 29 | Montreal | Mario Roberge | `0x0025BC` | 32 | 6 | 1 | 1 | 1 | 1 | 2 | 3 | L | 1 | 3 | 1 | 3 | 1 | 4 |
| 540 | 29 | Montreal | Todd Ewen | `0x002616` | 36 | 11 | 1 | 1 | 2 | 1 | 1 | 3 | R | 1 | 2 | 2 | 3 | 1 | 4 |
| 541 | 28 | Boston | Jim Wiemer | `0x000C10` | 36 | 10 | 2 | 1 | 2 | 2 | 1 | 2 | L | 2 | 0 | 2 | 3 | 2 | 4 |
| 542 | 27 | St. Louis | Bret Hedican | `0x0041A8` | 44 | 8 | 2 | 2 | 1 | 1 | 2 | 3 | L | 2 | 0 | 1 | 3 | 2 | 2 |
| 543 | 27 | Ottawa | Chris Luongo | `0x0032C2` | 23 | 6 | 2 | 2 | 1 | 3 | 1 | 1 | R | 2 | 0 | 2 | 4 | 2 | 3 |
| 544 | 27 | Calgary | Craig Berube | `0x0010AE` | 16 | 9 | 2 | 3 | 1 | 1 | 2 | 2 | L | 1 | 1 | 1 | 4 | 1 | 4 |
| 545 | 27 | Los Angeles | Tim Watters | `0x0020EC` | 5 | 6 | 2 | 2 | 1 | 2 | 1 | 2 | L | 2 | 0 | 2 | 1 | 2 | 2 |
| 546 | 26 | Los Angeles | Brent Thompson | `0x0020D4` | 3 | 5 | 2 | 2 | 1 | 2 | 1 | 2 | L | 2 | 0 | 2 | 1 | 1 | 4 |
| 547 | 26 | Ottawa | Gord Dineen | `0x0032EE` | 6 | 8 | 1 | 1 | 2 | 2 | 1 | 2 | R | 1 | 1 | 2 | 5 | 1 | 3 |
| 548 | 26 | Dallas | Mark Osiecki | `0x0023D4` | 23 | 9 | 2 | 2 | 1 | 2 | 2 | 2 | R | 2 | 0 | 1 | 3 | 1 | 2 |
| 549 | 26 | New York Islanders | Mick Vukota | `0x002C58` | 12 | 8 | 2 | 3 | 1 | 1 | 1 | 1 | R | 1 | 1 | 2 | 4 | 1 | 4 |
| 550 | 26 | Philadelphia | Shawn Cronin | `0x003608` | 44 | 10 | 1 | 1 | 1 | 1 | 1 | 2 | L | 1 | 3 | 1 | 5 | 1 | 3 |
| 551 | 25 | New York Islanders | Dennis Vaske | `0x002CFE` | 37 | 10 | 1 | 1 | 2 | 1 | 2 | 2 | L | 1 | 1 | 1 | 1 | 1 | 3 |
| 552 | 25 | Dallas | Enrico Ciccone | `0x0023FE` | 39 | 9 | 1 | 1 | 0 | 2 | 2 | 2 | L | 2 | 0 | 3 | 5 | 2 | 5 |
| 553 | 25 | New York Rangers | Mike Hartman | `0x002F56` | 18 | 7 | 2 | 3 | 1 | 0 | 1 | 3 | L | 1 | 1 | 1 | 5 | 1 | 4 |
| 554 | 24 | Tampa Bay | Stan Drulia | `0x004430` | 27 | 7 | 1 | 2 | 1 | 0 | 2 | 2 | R | 1 | 2 | 0 | 5 | 1 | 2 |
| 555 | 23 | Chicago | Adam Bennett | `0x0014E8` | 47 | 9 | 2 | 2 | 1 | 1 | 1 | 2 | R | 2 | 0 | 1 | 4 | 1 | 2 |
| 556 | 23 | Ottawa | Brad Marsh | `0x003304` | 14 | 11 | 2 | 2 | 0 | 3 | 1 | 3 | L | 1 | 0 | 2 | 5 | 2 | 2 |
| 557 | 23 | Pittsburgh | Bryan Fogarty | `0x0038F8` | 33 | 8 | 1 | 1 | 2 | 1 | 1 | 2 | L | 2 | 0 | 1 | 0 | 1 | 1 |
| 558 | 23 | San Jose | David Maley | `0x003DE4` | 25 | 8 | 1 | 1 | 1 | 1 | 2 | 3 | L | 2 | 0 | 1 | 4 | 1 | 4 |
| 559 | 23 | Washington | Jason Woolley | `0x00508A` | 25 | 6 | 2 | 2 | 1 | 1 | 1 | 2 | L | 2 | 0 | 1 | 2 | 1 | 2 |
| 560 | 22 | St. Louis | Kelly Chase | `0x00410A` | 39 | 8 | 1 | 1 | 1 | 2 | 1 | 1 | R | 1 | 1 | 2 | 3 | 1 | 5 |
| 561 | 21 | Anaheim | Bobby Dollas | `0x005524` | 32 | 11 | 2 | 1 | 1 | 2 | 1 | 2 | R | 1 | 0 | 2 | 1 | 1 | 1 |
| 562 | 21 | Anaheim | Dennis Vial | `0x00553A` | 17 | 11 | 1 | 1 | 1 | 2 | 1 | 3 | L | 1 | 0 | 2 | 1 | 1 | 3 |
| 563 | 21 | Toronto | Ken Baumgartnr | `0x00471A` | 22 | 9 | 2 | 2 | 0 | 2 | 1 | 2 | L | 1 | 0 | 2 | 3 | 2 | 4 |
| 564 | 21 | Anaheim | Stu Grimson | `0x005416` | 23 | 11 | 1 | 1 | 0 | 2 | 1 | 3 | L | 1 | 1 | 2 | 5 | 1 | 4 |
| 565 | 21 | Chicago | Stu Grimson | `0x0013AE` | 23 | 11 | 1 | 1 | 0 | 2 | 1 | 3 | L | 1 | 1 | 2 | 5 | 1 | 4 |
| 566 | 20 | Hartford | Mark Greig | `0x001D52` | 17 | 7 | 1 | 1 | 2 | 1 | 0 | 1 | R | 1 | 1 | 1 | 0 | 1 | 3 |
| 567 | 19 | Tampa Bay | Matt Hervey | `0x0044CC` | 26 | 9 | 1 | 1 | 2 | 1 | 1 | 2 | R | 1 | 0 | 1 | 3 | 0 | 4 |
| 568 | 18 | Tampa Bay | Chris Lipuma | `0x0044B6` | 40 | 6 | 1 | 1 | 2 | 1 | 1 | 0 | L | 1 | 0 | 1 | 1 | 1 | 4 |
| 569 | 17 | Quebec | Chris Simon | `0x003ACC` | 12 | 13 | 1 | 1 | 1 | 0 | 1 | 2 | L | 1 | 1 | 0 | 5 | 1 | 5 |
| 570 | 17 | Philadelphia | Dave Brown | `0x003554` | 21 | 9 | 1 | 1 | 0 | 2 | 1 | 2 | R | 1 | 0 | 2 | 5 | 1 | 3 |
| 571 | 17 | Pittsburgh | Jay Caufield | `0x00382A` | 16 | 14 | 1 | 2 | 0 | 1 | 1 | 3 | R | 1 | 0 | 1 | 3 | 1 | 4 |
| 572 | 15 | Quebec | Tony Twist | `0x003AE2` | 15 | 10 | 1 | 1 | 1 | 0 | 1 | 1 | L | 1 | 0 | 1 | 4 | 1 | 4 |
| 573 | 12 | New Jersey | Myles O'Connor | `0x002A0C` | 5 | 7 | 1 | 1 | 0 | 1 | 1 | 1 | L | 1 | 0 | 1 | 3 | 0 | 3 |
| 574 | 11 | Los Angeles | Rene Chapdlaine | `0x002102` | 8 | 8 | 1 | 1 | 0 | 1 | 0 | 1 | R | 1 | 0 | 1 | 3 | 1 | 3 |
| 575 | 9 | Calgary | Greg Smyth | `0x0011F2` | 6 | 10 | 0 | 0 | 1 | 0 | 0 | 1 | R | 0 | 1 | 1 | 4 | 0 | 4 |

## Goalies

Goalies share the packed record format, but several fields have different meanings. `Off`, `Chk`, `Stk`, and `ShA` are retained raw nibbles without a displayed goalie-rating meaning; they are included so every packed rating is represented. `Puck`, `GlvR`, `StkL`, `StkR`, and `GlvL` use the goalie mappings. `Hand` is the goalie glove hand decoded from the same low bit as skater handedness.

| Rank | Overall | Team | Player | ROM | # | Wt | Agl | Spd | Off* | Def | Puck | Chk* | Hand | Stk* | ShA* | StkR | StkL | GlvR | GlvL |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 77 | Chicago | Ed Belfour | `0x0012C6` | 30 | 6 | 6 | 4 | 6 | 6 | 5 | 0 | L | 0 | 0 | 6 | 6 | 5 | 5 |
| 2 | 76 | Montreal | Patrick Roy | `0x0024D0` | 33 | 6 | 6 | 4 | 4 | 4 | 6 | 0 | L | 0 | 0 | 5 | 5 | 6 | 6 |
| 3 | 64 | Buffalo | Grant Fuhr | `0x000CDE` | 31 | 7 | 5 | 4 | 5 | 5 | 5 | 0 | R | 0 | 0 | 4 | 4 | 5 | 5 |
| 4 | 58 | Toronto | Felix Potvin | `0x0045AA` | 29 | 6 | 4 | 4 | 6 | 6 | 4 | 0 | L | 0 | 0 | 4 | 4 | 4 | 4 |
| 5 | 57 | Winnipeg | Bob Essensa | `0x004B9C` | 35 | 3 | 4 | 4 | 5 | 5 | 4 | 0 | L | 0 | 0 | 4 | 4 | 4 | 4 |
| 6 | 57 | Pittsburgh | Tom Barrasso | `0x0036D8` | 35 | 10 | 4 | 4 | 5 | 5 | 4 | 0 | R | 0 | 0 | 4 | 4 | 4 | 4 |
| 7 | 56 | Edmonton | Bill Ranford | `0x0018DA` | 30 | 4 | 4 | 3 | 3 | 3 | 4 | 0 | L | 0 | 0 | 4 | 4 | 5 | 4 |
| 8 | 55 | Philadelphia | Tommy Soderstrom | `0x0033DA` | 30 | 3 | 4 | 4 | 5 | 5 | 4 | 0 | L | 0 | 0 | 3 | 4 | 3 | 4 |
| 9 | 54 | Vancouver | Kirk McLean | `0x0048AE` | 1 | 8 | 4 | 4 | 4 | 4 | 4 | 0 | L | 0 | 0 | 4 | 4 | 3 | 3 |
| 10 | 54 | Quebec | Ron Hextall | `0x0039CE` | 27 | 7 | 4 | 4 | 4 | 4 | 4 | 0 | L | 0 | 0 | 4 | 4 | 3 | 3 |
| 11 | 54 | Hartford | Sean Burke | `0x001BD6` | 1 | 10 | 4 | 3 | 2 | 2 | 4 | 0 | L | 0 | 0 | 4 | 4 | 4 | 4 |
| 12 | 53 | Calgary | Mike Vernon | `0x000FCC` | 30 | 4 | 3 | 4 | 4 | 4 | 4 | 0 | L | 0 | 0 | 4 | 4 | 3 | 3 |
| 13 | 52 | St. Louis | Curtis Joseph | `0x003FC2` | 31 | 6 | 4 | 4 | 6 | 6 | 4 | 0 | L | 0 | 0 | 3 | 3 | 4 | 4 |
| 14 | 51 | Detroit | Tim Cheveldae | `0x0015C0` | 32 | 6 | 4 | 4 | 4 | 4 | 4 | 0 | L | 0 | 0 | 4 | 3 | 4 | 4 |
| 15 | 50 | New York Rangers | Mike Richter | `0x002E08` | 35 | 7 | 3 | 4 | 4 | 4 | 3 | 0 | L | 0 | 0 | 4 | 4 | 4 | 4 |
| 16 | 49 | Florida | John Vanbiesbrk | `0x005178` | 34 | 5 | 3 | 4 | 5 | 5 | 3 | 0 | L | 0 | 0 | 3 | 4 | 3 | 4 |
| 17 | 49 | New York Rangers | John Vanbiesbrk | `0x002DEE` | 34 | 5 | 3 | 4 | 5 | 5 | 3 | 0 | L | 0 | 0 | 3 | 4 | 3 | 4 |
| 18 | 48 | Boston | Andy Moog | `0x0009F8` | 35 | 4 | 4 | 4 | 2 | 2 | 4 | 0 | L | 0 | 0 | 4 | 3 | 4 | 3 |
| 19 | 45 | Boston | John Blue | `0x000A0C` | 39 | 6 | 3 | 4 | 5 | 5 | 3 | 0 | L | 0 | 0 | 4 | 3 | 4 | 3 |
| 20 | 43 | New Jersey | Chris Terreri | `0x0027D2` | 31 | 2 | 4 | 4 | 4 | 4 | 3 | 0 | L | 0 | 0 | 3 | 3 | 3 | 3 |
| 21 | 41 | Washington | Don Beaupre | `0x004E8E` | 33 | 4 | 3 | 4 | 3 | 3 | 3 | 0 | L | 0 | 0 | 3 | 3 | 3 | 3 |
| 22 | 41 | Los Angeles | Kelly Hrudey | `0x001EE4` | 32 | 6 | 3 | 3 | 4 | 4 | 4 | 0 | L | 0 | 0 | 2 | 2 | 2 | 3 |
| 23 | 39 | San Jose | Arturs Irbe | `0x003CD4` | 32 | 5 | 2 | 4 | 4 | 4 | 3 | 0 | L | 0 | 0 | 3 | 3 | 2 | 2 |
| 24 | 39 | Philadelphia | Dominic Roussel | `0x0033F4` | 33 | 6 | 3 | 3 | 3 | 3 | 3 | 0 | L | 0 | 0 | 3 | 3 | 2 | 2 |
| 25 | 39 | Dallas | Jon Casey | `0x0021E0` | 30 | 2 | 4 | 3 | 4 | 4 | 3 | 0 | L | 0 | 0 | 2 | 2 | 3 | 4 |
| 26 | 36 | Vancouver | Kay Whitmore | `0x0048C4` | 35 | 5 | 2 | 3 | 5 | 5 | 2 | 0 | L | 0 | 0 | 2 | 3 | 2 | 3 |
| 27 | 36 | Tampa Bay | Wendell Young | `0x0042A8` | 1 | 6 | 3 | 3 | 2 | 2 | 3 | 0 | L | 0 | 0 | 3 | 2 | 3 | 3 |
| 28 | 35 | Los Angeles | Robb Stauber | `0x001EFA` | 35 | 6 | 2 | 3 | 5 | 5 | 3 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 29 | 34 | Dallas | Darcy Wakaluk | `0x0021F4` | 35 | 6 | 2 | 3 | 3 | 3 | 3 | 0 | L | 0 | 0 | 2 | 2 | 3 | 2 |
| 30 | 34 | Toronto | Daren Puppa | `0x0045C0` | 1 | 9 | 3 | 4 | 5 | 5 | 2 | 0 | R | 0 | 0 | 2 | 2 | 3 | 3 |
| 31 | 33 | Montreal | Andre Racicot | `0x0024E6` | 37 | 4 | 2 | 3 | 3 | 3 | 3 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 32 | 32 | Buffalo | Dominik Hasek | `0x000D06` | 39 | 4 | 3 | 3 | 5 | 5 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 33 | 32 | Hartford | Frank Pietrngelo | `0x001C02` | 40 | 6 | 3 | 3 | 1 | 1 | 3 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 34 | 32 | Calgary | Jeff Reese | `0x000FE2` | 35 | 4 | 2 | 3 | 4 | 4 | 2 | 0 | L | 0 | 0 | 2 | 2 | 3 | 3 |
| 35 | 32 | Chicago | Jim Waite | `0x0012DA` | 29 | 6 | 3 | 3 | 3 | 3 | 2 | 0 | L | 0 | 0 | 2 | 2 | 3 | 3 |
| 36 | 32 | Pittsburgh | Ken Wregget | `0x0036EE` | 31 | 8 | 2 | 3 | 4 | 4 | 2 | 0 | L | 0 | 0 | 3 | 2 | 3 | 2 |
| 37 | 32 | Florida | Mark Fitzpatrik | `0x005192` | 30 | 7 | 2 | 4 | 2 | 2 | 3 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 38 | 32 | New York Islanders | Mark Fitzpatrik | `0x002B00` | 30 | 7 | 2 | 4 | 2 | 2 | 3 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 39 | 32 | Washington | Rick Tabaracci | `0x004EA4` | 31 | 6 | 2 | 3 | 1 | 1 | 2 | 0 | L | 0 | 0 | 3 | 3 | 2 | 2 |
| 40 | 31 | Detroit | Vincent Riendeau | `0x0015D8` | 37 | 6 | 3 | 3 | 2 | 2 | 2 | 0 | L | 0 | 0 | 3 | 2 | 2 | 3 |
| 41 | 30 | New York Islanders | Glenn Healy | `0x002AEA` | 35 | 5 | 2 | 4 | 4 | 4 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 42 | 30 | Quebec | Stephane Fiset | `0x0039E4` | 35 | 5 | 3 | 4 | 3 | 3 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 43 | 29 | Anaheim | Guy Hebert | `0x0053C2` | 29 | 6 | 2 | 3 | 3 | 3 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 44 | 29 | St. Louis | Guy Hebert | `0x003FDA` | 29 | 6 | 2 | 3 | 3 | 3 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 45 | 29 | Buffalo | Tom Draper | `0x000CF2` | 35 | 6 | 2 | 3 | 3 | 3 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 46 | 28 | Ottawa | Daniel Berthiaume | `0x003100` | 32 | 1 | 3 | 3 | 1 | 1 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 47 | 28 | Winnipeg | Jim Hrivnak | `0x004BB2` | 30 | 6 | 2 | 3 | 2 | 2 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 48 | 28 | Tampa Bay | Pat Jablonski | `0x0042C0` | 35 | 5 | 2 | 3 | 2 | 2 | 2 | 0 | R | 0 | 0 | 2 | 2 | 2 | 2 |
| 49 | 28 | Los Angeles | Rick Knickle | `0x001F10` | 1 | 2 | 2 | 3 | 3 | 3 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 1 |
| 50 | 27 | San Jose | Jeff Hackett | `0x003CEA` | 30 | 5 | 2 | 3 | 1 | 1 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 51 | 27 | Ottawa | Peter Sidorkwicz | `0x0030E6` | 31 | 6 | 2 | 3 | 1 | 1 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 52 | 26 | San Jose | Brian Hayward | `0x003D00` | 1 | 6 | 2 | 3 | 0 | 0 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 53 | 26 | Toronto | Rick Wamsley | `0x0045D6` | 30 | 6 | 2 | 3 | 0 | 0 | 2 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 54 | 24 | New Jersey | Craig Billington | `0x0027EA` | 1 | 4 | 3 | 4 | 2 | 2 | 1 | 0 | L | 0 | 0 | 2 | 2 | 2 | 2 |
| 55 | 24 | Hartford | Mario Gosselin | `0x001BEA` | 31 | 3 | 2 | 3 | 4 | 4 | 2 | 0 | L | 0 | 0 | 1 | 1 | 2 | 2 |
| 56 | 21 | Philadelphia | Steph Beauregard | `0x00340E` | 35 | 7 | 2 | 3 | 0 | 0 | 1 | 0 | R | 0 | 0 | 2 | 2 | 2 | 2 |
| 57 | 20 | Tampa Bay | J.C. Bergeron | `0x0042D8` | 30 | 7 | 2 | 3 | 2 | 2 | 2 | 0 | L | 0 | 0 | 1 | 1 | 1 | 1 |
| 58 | 17 | Anaheim | Ron Tugnutt | `0x0053D6` | 1 | 2 | 2 | 3 | 3 | 3 | 1 | 0 | L | 0 | 0 | 1 | 1 | 1 | 2 |
| 59 | 17 | Edmonton | Ron Tugnutt | `0x0018F0` | 1 | 2 | 2 | 3 | 3 | 3 | 1 | 0 | L | 0 | 0 | 1 | 1 | 1 | 2 |
