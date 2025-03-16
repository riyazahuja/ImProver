/-- Proves that when a `SeminormedAddGroup` structure is constructed from an
`AddGroupSeminormClass` that satisfies `IsNonarchimedean`, the group has an `IsUltrametricDist`. -/
lemma AddGroupSeminormClass.isUltrametricDist [AddGroup α] [AddGroupSeminormClass F α ℝ]
    [inst : Dist α] {f : F} (hna : IsNonarchimedean f)
    (hd : inst = (AddGroupSeminormClass.toSeminormedAddGroup f).toDist := by rfl) :
    IsUltrametricDist α :=
  ⟨fun x y z ↦ by simpa only [hd, dist_eq_norm, AddGroupSeminormClass.toSeminormedAddGroup_norm_eq,
      ← sub_add_sub_cancel x y z] using hna _ _⟩

