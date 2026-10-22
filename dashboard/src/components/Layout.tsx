import { NavLink, Outlet } from "react-router-dom"

const links = [
  { to: "/", label: "Overview" },
  { to: "/clients", label: "Federated clients" },
  { to: "/privacy", label: "Privacy & security" },
  { to: "/evaluation", label: "Model evaluation" },
  { to: "/experiments", label: "Experiments" },
]

export function Layout() {
  return (
    <div className="min-h-screen bg-canvas text-ink">
      <aside className="fixed inset-y-0 left-0 hidden w-60 border-r border-line bg-panel-2 px-4 py-6 md:block">
        <p className="text-xs uppercase tracking-[0.2em] text-accent">Federated IoT</p>
        <h1 className="mt-1 text-lg font-semibold">Privacy dashboard</h1>
        <p className="mt-2 text-xs leading-5 text-muted">
          Research monitor for Flower + Opacus + SecAgg+. Not a production IDS console.
        </p>
        <nav className="mt-8 flex flex-col gap-1">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm ${
                  isActive
                    ? "bg-panel text-accent"
                    : "text-muted hover:bg-panel hover:text-ink"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="md:pl-60">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-canvas/90 px-4 py-3 backdrop-blur md:px-8">
          <div className="flex gap-3 overflow-x-auto md:hidden">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === "/"}
                className={({ isActive }) =>
                  `whitespace-nowrap text-sm ${isActive ? "text-accent" : "text-muted"}`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>
          <p className="hidden text-sm text-muted md:block">
            UNSW-NB15 · 5 simulated clients · MLP · Flower App
          </p>
        </header>
        <main className="px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
