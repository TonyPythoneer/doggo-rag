import type { DatasetImage } from "./types/pipeline";

export const SPECIES_HEADINGS: Record<string, string> = {
  bear: "Bears",
  cat: "Cats",
  coyote: "Coyotes",
  dingo: "Dingoes",
  dog: "Dogs",
  fox: "Foxes",
  hyena: "Hyenas",
  koala: "Koalas",
  raccoon: "Raccoons",
  red_panda: "Red pandas",
  wolf: "Wolves",
};

export function speciesHeading(species: string): string {
  return SPECIES_HEADINGS[species] ?? species.replace(/_/g, " ");
}

/** Dogs first, then the other species alphabetically by heading. */
export function groupBySpecies(images: DatasetImage[]): [string, DatasetImage[]][] {
  const groups = new Map<string, DatasetImage[]>();
  for (const image of images) {
    const group = groups.get(image.species);
    if (group) group.push(image);
    else groups.set(image.species, [image]);
  }
  return Array.from(groups.entries()).sort(([a], [b]) => {
    if (a === "dog") return -1;
    if (b === "dog") return 1;
    return speciesHeading(a).localeCompare(speciesHeading(b));
  });
}

export function dogLabel(isDog: boolean | null): string {
  if (isDog === null) return "Uncertain";
  return isDog ? "Domestic dog" : "Not a domestic dog";
}
