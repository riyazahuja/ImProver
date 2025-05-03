/-- An alias for `Type max u v`, to deal around unification issues. -/
@[nolint checkUnivs, to_additive existing TypeMax]
abbrev TypeMax.{u, v} := Type max u v

