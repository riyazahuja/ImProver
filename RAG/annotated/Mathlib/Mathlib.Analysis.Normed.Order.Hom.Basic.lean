/-- Constructs a `SeminormedGroup` structure from a `GroupSeminormClass` on a `Group`. -/
-- See note [reducible non-instances]
@[to_additive "Constructs a `SeminormedAddGroup` structure from an `AddGroupSeminormClass` on an
`AddGroup`."]
abbrev GroupSeminormClass.toSeminormedGroup [Group α] [GroupSeminormClass F α ℝ]
    (f : F) : SeminormedGroup α where
  norm := f
  dist x y := f (x / y)
  dist_eq _ _ := rfl
                    /-
                      F : Type u_1
                      α : Type u_2
                      inst✝² : FunLike F α Real
                      inst✝¹ : Group α
                      inst✝ : GroupSeminormClass F α Real
                      f : F
                      x✝ : α
                      ⊢ Eq (Dist.dist x✝ x✝) 0
                    -/
  dist_self _ := by simp
                    /-
                      🎉 no goals
                    -/
                      /-
                        F : Type u_1
                        α : Type u_2
                        inst✝² : FunLike F α Real
                        inst✝¹ : Group α
                        inst✝ : GroupSeminormClass F α Real
                        f : F
                        x y : α
                        ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                      -/
  dist_comm x y := by simp only [← map_inv_eq_map f (x / y), inv_div]
                      /-
                        🎉 no goals
                      -/
                            /-
                              F : Type u_1
                              α : Type u_2
                              inst✝² : FunLike F α Real
                              inst✝¹ : Group α
                              inst✝ : GroupSeminormClass F α Real
                              f : F
                              x y z : α
                              ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
                            -/
  dist_triangle x y z := by simpa using map_mul_le_add f (x / y) (y / z)
                            /-
                              🎉 no goals
                            -/


@[to_additive]
lemma GroupSeminormClass.toSeminormedGroup_norm_eq [Group α] [GroupSeminormClass F α ℝ]
    (f : F) (x : α) : @norm _ (GroupSeminormClass.toSeminormedGroup f).toNorm x = f x := rfl


/-- Constructs a `SeminormedCommGroup` structure from a `GroupSeminormClass` on a `CommGroup`. -/
-- See note [reducible non-instances]
@[to_additive "Constructs a `SeminormedAddCommGroup` structure from an `AddGroupSeminormClass` on an
`AddCommGroup`."]
abbrev GroupSeminormClass.toSeminormedCommGroup [CommGroup α] [GroupSeminormClass F α ℝ]
    (f : F) : SeminormedCommGroup α where
  __ := GroupSeminormClass.toSeminormedGroup f
  __ : CommGroup α := inferInstance


@[to_additive]
lemma GroupSeminormClass.toSeminormedCommGroup_norm_eq [CommGroup α] [GroupSeminormClass F α ℝ]
    (f : F) (x : α) : @norm _ (GroupSeminormClass.toSeminormedCommGroup f).toNorm x = f x := rfl


/-- Constructs a `NormedGroup` structure from a `GroupNormClass` on a `Group`. -/
-- See note [reducible non-instances]
@[to_additive "Constructs a `NormedAddGroup` structure from an `AddGroupNormClass` on an
`AddGroup`."]
abbrev GroupNormClass.toNormedGroup [Group α] [GroupNormClass F α ℝ]
    (f : F) : NormedGroup α where
  __ := GroupSeminormClass.toSeminormedGroup f
  eq_of_dist_eq_zero h := div_eq_one.mp (eq_one_of_map_eq_zero f h)


@[to_additive]
lemma GroupNormClass.toNormedGroup_norm_eq [Group α] [GroupNormClass F α ℝ]
    (f : F) (x : α) : @norm _ (GroupNormClass.toNormedGroup f).toNorm x = f x := rfl


/-- Constructs a `NormedCommGroup` structure from a `GroupNormClass` on a `CommGroup`. -/
-- See note [reducible non-instances]
@[to_additive "Constructs a `NormedAddCommGroup` structure from an `AddGroupNormClass` on an
`AddCommGroup`."]
abbrev GroupNormClass.toNormedCommGroup [CommGroup α] [GroupNormClass F α ℝ]
    (f : F) : NormedCommGroup α where
  __ := GroupNormClass.toNormedGroup f
  __ : CommGroup α := inferInstance


@[to_additive]
lemma GroupNormClass.toNormedCommGroup_norm_eq [CommGroup α] [GroupNormClass F α ℝ]
    (f : F) (x : α) : @norm _ (GroupNormClass.toNormedCommGroup f).toNorm x = f x := rfl

