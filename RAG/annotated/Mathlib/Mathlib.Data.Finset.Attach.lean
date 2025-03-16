/-- `attach s` takes the elements of `s` and forms a new set of elements of the subtype
`{x // x ∈ s}`. -/
def attach (s : Finset α) : Finset { x // x ∈ s } :=
  ⟨Multiset.attach s.1, nodup_attach.2 s.2⟩


@[simp]
theorem attach_val (s : Finset α) : s.attach.1 = s.1.attach :=
  rfl


@[simp]
theorem mem_attach (s : Finset α) : ∀ x, x ∈ s.attach :=
  Multiset.mem_attach _


