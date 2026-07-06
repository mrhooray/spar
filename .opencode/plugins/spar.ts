import { realpathSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const sparRoot = resolve(realpathSync(dirname(fileURLToPath(import.meta.url))), "../..")

export default async () => ({
  "shell.env": async (_input: unknown, output: { env: Record<string, string> }) => {
    output.env.SPAR_ROOT = sparRoot
  },
})
