import { AVAILABLE_BOTS } from "./bot";

const list = document.getElementById("bot-list");
if (!(list instanceof HTMLTableSectionElement)) throw new Error("Missing bot catalogue table.");

for (const bot of AVAILABLE_BOTS) {
  const row = document.createElement("tr");
  const name = document.createElement("th");
  name.scope = "row";
  name.textContent = bot.label;
  const description = document.createElement("td");
  description.textContent = bot.description;
  row.append(name, description);
  list.append(row);
}
