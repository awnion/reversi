# Reversi

A browser-based [Reversi](https://en.wikipedia.org/wiki/Reversi) (Othello) game built with TypeScript.

## Structure

Bun workspace monorepo:

- `packages/core` — game rules and logic
- `packages/board-ui` — DOM-based board renderer
- `apps/web` — playable web app

## Quick start

```sh
bun install
bun dev
```

Open `http://localhost:5173` in your browser.

## Build

```sh
bun run build
```

## Test

```sh
# unit tests (core logic)
bun run test:unit

# browser integration tests (vitest + playwright)
bun run test:browser

# e2e tests (requires browsers: bun run install:browsers)
bun run test:e2e
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
