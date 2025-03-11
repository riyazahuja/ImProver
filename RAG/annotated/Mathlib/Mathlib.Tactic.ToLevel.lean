/-- A class to create `Level` expressions that denote particular universe levels in Lean.
`Lean.ToLevel.toLevel.{u}` evaluates to a `Lean.Level` term representing `u` -/
@[pp_with_univ]
class ToLevel.{u} where
  /-- A `Level` that represents the universe level `u`. -/
  toLevel : Level
  /-- The universe itself. This is only here to avoid the "unused universe parameter" error. -/
  univ : Type u := Sort u

instance : ToLevel.{0} where
  toLevel := .zero


instance [ToLevel.{u}] : ToLevel.{u+1} where
  toLevel := .succ toLevel.{u}


/-- `ToLevel` for `max u v`. This is not an instance since it causes divergence. -/
def ToLevel.max [ToLevel.{u}] [ToLevel.{v}] : ToLevel.{max u v} where
  toLevel := .max toLevel.{u} toLevel.{v}


/-- `ToLevel` for `imax u v`. This is not an instance since it causes divergence. -/
def ToLevel.imax [ToLevel.{u}] [ToLevel.{v}] : ToLevel.{imax u v} where
  toLevel := .imax toLevel.{u} toLevel.{v}


