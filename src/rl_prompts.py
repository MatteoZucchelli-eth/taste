"""Step 2 prompt set (CONTEXT §6g decision 4): small, hand-made, generic art/design prompts (subjects x media/styles).
Half name only a subject (the actor is free to choose the style, which is where taste shows); half also fix a medium.
TRAIN (20) is used for RL; EVAL (10) is never trained on and is used for every comparison and the human eval."""

TRAIN = [
    "a portrait of a woman",
    "a still life with fruit and a jug",
    "a landscape with hills and a river",
    "a city street at night",
    "an abstract composition",
    "a vase of flowers",
    "a couple dancing",
    "a harbour with boats",
    "a figure reading in a room",
    "a forest in autumn",
    "a portrait of an old man, oil painting",
    "a seascape with a storm, watercolor",
    "geometric shapes and lines, print",
    "a horse in a field, ink drawing",
    "a crowded market, gouache painting",
    "a cat sleeping on a chair, pastel drawing",
    "a mountain village, woodcut",
    "a nude figure study, charcoal drawing",
    "an interior with a window, acrylic painting",
    "a garden in bloom, mosaic",
]

EVAL = [
    "a portrait of a young man",
    "a bowl of lemons on a table",
    "a lake at sunrise",
    "a cathedral",
    "an abstract painting with bold colors",
    "musicians playing in a cafe",
    "a bridge over a canal, oil painting",
    "a woman with an umbrella in the rain, watercolor",
    "a bird on a branch, ink drawing",
    "a factory by a river, linocut",
]
