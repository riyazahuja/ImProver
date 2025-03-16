/-- Every topological group in which there exists a compact set with nonempty interior
is locally compact. -/
@[to_additive
  "Every topological additive group
  in which there exists a compact set with nonempty interior is locally compact."]
theorem TopologicalSpace.PositiveCompacts.locallyCompactSpace_of_group
    (K : PositiveCompacts G) : LocallyCompactSpace G :=
  let ⟨_x, hx⟩ := K.interior_nonempty
  K.isCompact.locallyCompactSpace_of_mem_nhds_of_group (mem_interior_iff_mem_nhds.1 hx)

