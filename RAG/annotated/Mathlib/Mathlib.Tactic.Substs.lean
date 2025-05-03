/--
Applies the `subst` tactic to all given hypotheses from left to right.
-/
syntax (name := substs) "substs" (colGt ppSpace ident)* : tactic


macro_rules
| `(tactic| substs $xs:ident*) => `(tactic| ($[subst $xs]*))


