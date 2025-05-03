private abbrev ProdSpaceFun : Type _ :=
  (X ⊕ Y) × (X ⊕ Y) → ℝ


private abbrev Cb : Type _ :=
  BoundedContinuousFunction ((X ⊕ Y) × (X ⊕ Y)) ℝ


private def maxVar : ℝ≥0 :=
  2 * ⟨diam (univ : Set X), diam_nonneg⟩ + 1 + 2 * ⟨diam (univ : Set Y), diam_nonneg⟩


private theorem one_le_maxVar : 1 ≤ maxVar X Y :=
  calc
                                         /-
                                           X : Type u
                                           Y : Type v
                                           inst✝¹ : MetricSpace X
                                           inst✝ : MetricSpace Y
                                           ⊢ Eq 1 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 0) 1) (HMul.hMul 2 0))
                                         -/
    (1 : Real) = 2 * 0 + 1 + 2 * 0 := by simp
                                         /-
                                           🎉 no goals
                                         -/
                                                                    /-
                                                                      X : Type u
                                                                      Y : Type v
                                                                      inst✝¹ : MetricSpace X
                                                                      inst✝ : MetricSpace Y
                                                                      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 0) 1) (HMul.hMul 2 0)) (HAdd.hAdd ( …
                                                                    -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    _ ≤ 2 * diam (univ : Set X) + 1 + 2 * diam (univ : Set Y) := by gcongr <;> positivity
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The set of functions on `X ⊕ Y` that are candidates distances to realize the
minimum of the Hausdorff distances between `X` and `Y` in a coupling. -/
def candidates : Set (ProdSpaceFun X Y) :=
  { f | (((((∀ x y : X, f (Sum.inl x, Sum.inl y) = dist x y) ∧
      ∀ x y : Y, f (Sum.inr x, Sum.inr y) = dist x y) ∧
      ∀ x y, f (x, y) = f (y, x)) ∧
      ∀ x y z, f (x, z) ≤ f (x, y) + f (y, z)) ∧
      ∀ x, f (x, x) = 0) ∧
      ∀ x y, f (x, y) ≤ maxVar X Y }


/-- Version of the set of candidates in bounded_continuous_functions, to apply Arzela-Ascoli. -/
private def candidatesB : Set (Cb X Y) :=
  { f : Cb X Y | (f : _ → ℝ) ∈ candidates X Y }


private theorem maxVar_bound [CompactSpace X] [Nonempty X] [CompactSpace Y] [Nonempty Y] :
    dist x y ≤ maxVar X Y :=
  calc
    dist x y ≤ diam (univ : Set (X ⊕ Y)) :=
      dist_le_diam_of_mem isBounded_of_compactSpace (mem_univ _) (mem_univ _)
                                                         /-
                                                           X : Type u
                                                           Y : Type v
                                                           inst✝⁵ : MetricSpace X
                                                           inst✝⁴ : MetricSpace Y
                                                           x y : Sum X Y
                                                           inst✝³ : CompactSpace X
                                                           inst✝² : Nonempty X
                                                           inst✝¹ : CompactSpace Y
                                                           inst✝ : Nonempty Y
                                                           ⊢ Eq (Metric.diam Set.univ) (Metric.diam (Union.union (Set.range Sum.inl) (Set …
                                                         -/
    _ = diam (range inl ∪ range inr : Set (X ⊕ Y)) := by rw [range_inl_union_range_inr]
                                                         /-
                                                           🎉 no goals
                                                         -/
    _ ≤ diam (range inl : Set (X ⊕ Y)) + dist (inl default) (inr default) +
        diam (range inr : Set (X ⊕ Y)) :=
      (diam_union (mem_range_self _) (mem_range_self _))
    _ = diam (univ : Set X) + (dist (α := X) default default + 1 + dist (α := Y) default default) +
        diam (univ : Set Y) := by
      /-
        X : Type u
        Y : Type v
        inst✝⁵ : MetricSpace X
        inst✝⁴ : MetricSpace Y
        x y : Sum X Y
        inst✝³ : CompactSpace X
        inst✝² : Nonempty X
        inst✝¹ : CompactSpace Y
        inst✝ : Nonempty Y
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Metric.diam (Set.range Sum.inl)) (Dist.dist (Sum.i …
      -/
      rw [isometry_inl.diam_range, isometry_inr.diam_range]
      /-
        X : Type u
        Y : Type v
        inst✝⁵ : MetricSpace X
        inst✝⁴ : MetricSpace Y
        x y : Sum X Y
        inst✝³ : CompactSpace X
        inst✝² : Nonempty X
        inst✝¹ : CompactSpace Y
        inst✝ : Nonempty Y
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Metric.diam Set.univ) (Dist.dist (Sum.inl Inhabite …
      -/
      rfl
      /-
        🎉 no goals
      -/
                                                                    /-
                                                                      X : Type u
                                                                      Y : Type v
                                                                      inst✝⁵ : MetricSpace X
                                                                      inst✝⁴ : MetricSpace Y
                                                                      x y : Sum X Y
                                                                      inst✝³ : CompactSpace X
                                                                      inst✝² : Nonempty X
                                                                      inst✝¹ : CompactSpace Y
                                                                      inst✝ : Nonempty Y
                                                                      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Metric.diam Set.univ) (HAdd.hAdd (HAdd.hAdd (Dist. …
                                                                    -/
    _ = 1 * diam (univ : Set X) + 1 + 1 * diam (univ : Set Y) := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      X : Type u
                                                                      Y : Type v
                                                                      inst✝⁵ : MetricSpace X
                                                                      inst✝⁴ : MetricSpace Y
                                                                      x y : Sum X Y
                                                                      inst✝³ : CompactSpace X
                                                                      inst✝² : Nonempty X
                                                                      inst✝¹ : CompactSpace Y
                                                                      inst✝ : Nonempty Y
                                                                      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (Metric.diam Set.univ)) 1) (HMul.hM …
                                                                    -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    _ ≤ 2 * diam (univ : Set X) + 1 + 2 * diam (univ : Set Y) := by gcongr <;> norm_num
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


private theorem candidates_symm (fA : f ∈ candidates X Y) : f (x, y) = f (y, x) :=
  fA.1.1.1.2 x y


private theorem candidates_triangle (fA : f ∈ candidates X Y) : f (x, z) ≤ f (x, y) + f (y, z) :=
  fA.1.1.2 x y z


private theorem candidates_refl (fA : f ∈ candidates X Y) : f (x, x) = 0 :=
  fA.1.2 x


private theorem candidates_nonneg (fA : f ∈ candidates X Y) : 0 ≤ f (x, y) := by
  have : 0 ≤ 2 * f (x, y) :=
    calc
      0 = f (x, x) := (candidates_refl fA).symm
      _ ≤ f (x, y) + f (y, x) := candidates_triangle fA
      _ = f (x, y) + f (x, y) := by rw [candidates_symm fA]
      _ = 2 * f (x, y) := by ring
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    x y : Sum X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    this : LE.le 0 (HMul.hMul 2 (f { fst := x, snd := y }))
    ⊢ LE.le 0 (f { fst := x, snd := y })
  -/
  linarith
  /-
    🎉 no goals
  -/


private theorem candidates_dist_inl (fA : f ∈ candidates X Y) (x y : X) :
    f (inl x, inl y) = dist x y :=
  fA.1.1.1.1.1 x y


private theorem candidates_dist_inr (fA : f ∈ candidates X Y) (x y : Y) :
    f (inr x, inr y) = dist x y :=
  fA.1.1.1.1.2 x y


private theorem candidates_le_maxVar (fA : f ∈ candidates X Y) : f (x, y) ≤ maxVar X Y :=
  fA.2 x y


/-- candidates are bounded by `maxVar X Y` -/
private theorem candidates_dist_bound (fA : f ∈ candidates X Y) :
    ∀ {x y : X ⊕ Y}, f (x, y) ≤ maxVar X Y * dist x y
  | inl x, inl y =>
    calc
      f (inl x, inl y) = dist x y := candidates_dist_inl fA x y
      _ = dist (α := X ⊕ Y) (inl x) (inl y) := by
        /-
          X : Type u
          Y : Type v
          inst✝¹ : MetricSpace X
          inst✝ : MetricSpace Y
          f : GromovHausdorff.ProdSpaceFun X Y
          fA : Membership.mem (GromovHausdorff.candidates X Y) f
          x y : X
          ⊢ Eq (Dist.dist x y) (Dist.dist (Sum.inl x) (Sum.inl y))
        -/
        rw [@Sum.dist_eq X Y]
        /-
          X : Type u
          Y : Type v
          inst✝¹ : MetricSpace X
          inst✝ : MetricSpace Y
          f : GromovHausdorff.ProdSpaceFun X Y
          fA : Membership.mem (GromovHausdorff.candidates X Y) f
          x y : X
          ⊢ Eq (Dist.dist x y) (Metric.Sum.dist (Sum.inl x) (Sum.inl y))
        -/
        rfl
        /-
          🎉 no goals
        -/
                                                      /-
                                                        X : Type u
                                                        Y : Type v
                                                        inst✝¹ : MetricSpace X
                                                        inst✝ : MetricSpace Y
                                                        f : GromovHausdorff.ProdSpaceFun X Y
                                                        fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                        x y : X
                                                        ⊢ Eq (Dist.dist (Sum.inl x) (Sum.inl y)) (HMul.hMul 1 (Dist.dist (Sum.inl x) ( …
                                                      -/
      _ = 1 * dist (α := X ⊕ Y) (inl x) (inl y) := by ring
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                  /-
                                                    X : Type u
                                                    Y : Type v
                                                    inst✝¹ : MetricSpace X
                                                    inst✝ : MetricSpace Y
                                                    f : GromovHausdorff.ProdSpaceFun X Y
                                                    fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                    x y : X
                                                    ⊢ LE.le (HMul.hMul 1 (Dist.dist (Sum.inl x) (Sum.inl y))) (HMul.hMul (↑(Gromov …
                                                  -/
      _ ≤ maxVar X Y * dist (inl x) (inl y) := by gcongr; exact one_le_maxVar X Y
                                                          /-
                                                            🎉 no goals
                                                          -/
  | inl x, inr y =>
    calc
      f (inl x, inr y) ≤ maxVar X Y := candidates_le_maxVar fA
                               /-
                                 X : Type u
                                 Y : Type v
                                 inst✝¹ : MetricSpace X
                                 inst✝ : MetricSpace Y
                                 f : GromovHausdorff.ProdSpaceFun X Y
                                 fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                 x : X
                                 y : Y
                                 ⊢ Eq (↑(GromovHausdorff.maxVar X Y)) (HMul.hMul (↑(GromovHausdorff.maxVar X Y) …
                               -/
      _ = maxVar X Y * 1 := by simp
                               /-
                                 🎉 no goals
                               -/
                                                  /-
                                                    X : Type u
                                                    Y : Type v
                                                    inst✝¹ : MetricSpace X
                                                    inst✝ : MetricSpace Y
                                                    f : GromovHausdorff.ProdSpaceFun X Y
                                                    fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                    x : X
                                                    y : Y
                                                    ⊢ LE.le (HMul.hMul (↑(GromovHausdorff.maxVar X Y)) 1) (HMul.hMul (↑(GromovHaus …
                                                  -/
      _ ≤ maxVar X Y * dist (inl x) (inr y) := by gcongr; apply Sum.one_le_dist_inl_inr
                                                          /-
                                                            🎉 no goals
                                                          -/
  | inr x, inl y =>
    calc
      f (inr x, inl y) ≤ maxVar X Y := candidates_le_maxVar fA
                               /-
                                 X : Type u
                                 Y : Type v
                                 inst✝¹ : MetricSpace X
                                 inst✝ : MetricSpace Y
                                 f : GromovHausdorff.ProdSpaceFun X Y
                                 fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                 x : Y
                                 y : X
                                 ⊢ Eq (↑(GromovHausdorff.maxVar X Y)) (HMul.hMul (↑(GromovHausdorff.maxVar X Y) …
                               -/
      _ = maxVar X Y * 1 := by simp
                               /-
                                 🎉 no goals
                               -/
                                                  /-
                                                    X : Type u
                                                    Y : Type v
                                                    inst✝¹ : MetricSpace X
                                                    inst✝ : MetricSpace Y
                                                    f : GromovHausdorff.ProdSpaceFun X Y
                                                    fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                    x : Y
                                                    y : X
                                                    ⊢ LE.le (HMul.hMul (↑(GromovHausdorff.maxVar X Y)) 1) (HMul.hMul (↑(GromovHaus …
                                                  -/
      _ ≤ maxVar X Y * dist (inl x) (inr y) := by gcongr; apply Sum.one_le_dist_inl_inr
                                                          /-
                                                            🎉 no goals
                                                          -/
  | inr x, inr y =>
    calc
      f (inr x, inr y) = dist x y := candidates_dist_inr fA x y
      _ = dist (α := X ⊕ Y) (inr x) (inr y) := by
        /-
          X : Type u
          Y : Type v
          inst✝¹ : MetricSpace X
          inst✝ : MetricSpace Y
          f : GromovHausdorff.ProdSpaceFun X Y
          fA : Membership.mem (GromovHausdorff.candidates X Y) f
          x y : Y
          ⊢ Eq (Dist.dist x y) (Dist.dist (Sum.inr x) (Sum.inr y))
        -/
        rw [@Sum.dist_eq X Y]
        /-
          X : Type u
          Y : Type v
          inst✝¹ : MetricSpace X
          inst✝ : MetricSpace Y
          f : GromovHausdorff.ProdSpaceFun X Y
          fA : Membership.mem (GromovHausdorff.candidates X Y) f
          x y : Y
          ⊢ Eq (Dist.dist x y) (Metric.Sum.dist (Sum.inr x) (Sum.inr y))
        -/
        rfl
        /-
          🎉 no goals
        -/
                                                      /-
                                                        X : Type u
                                                        Y : Type v
                                                        inst✝¹ : MetricSpace X
                                                        inst✝ : MetricSpace Y
                                                        f : GromovHausdorff.ProdSpaceFun X Y
                                                        fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                        x y : Y
                                                        ⊢ Eq (Dist.dist (Sum.inr x) (Sum.inr y)) (HMul.hMul 1 (Dist.dist (Sum.inr x) ( …
                                                      -/
      _ = 1 * dist (α := X ⊕ Y) (inr x) (inr y) := by ring
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                  /-
                                                    X : Type u
                                                    Y : Type v
                                                    inst✝¹ : MetricSpace X
                                                    inst✝ : MetricSpace Y
                                                    f : GromovHausdorff.ProdSpaceFun X Y
                                                    fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                    x y : Y
                                                    ⊢ LE.le (HMul.hMul 1 (Dist.dist (Sum.inr x) (Sum.inr y))) (HMul.hMul (↑(Gromov …
                                                  -/
      _ ≤ maxVar X Y * dist (inr x) (inr y) := by gcongr; exact one_le_maxVar X Y
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Technical lemma to prove that candidates are Lipschitz -/
private theorem candidates_lipschitz_aux (fA : f ∈ candidates X Y) :
    f (x, y) - f (z, t) ≤ 2 * maxVar X Y * dist (x, y) (z, t) :=
  calc
                                                               /-
                                                                 X : Type u
                                                                 Y : Type v
                                                                 inst✝¹ : MetricSpace X
                                                                 inst✝ : MetricSpace Y
                                                                 f : GromovHausdorff.ProdSpaceFun X Y
                                                                 x y z t : Sum X Y
                                                                 fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                                 ⊢ LE.le (HSub.hSub (f { fst := x, snd := y }) (f { fst := z, snd := t })) (HSu …
                                                               -/
    f (x, y) - f (z, t) ≤ f (x, t) + f (t, y) - f (z, t) := by gcongr; exact candidates_triangle fA
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                        /-
                                                          X : Type u
                                                          Y : Type v
                                                          inst✝¹ : MetricSpace X
                                                          inst✝ : MetricSpace Y
                                                          f : GromovHausdorff.ProdSpaceFun X Y
                                                          x y z t : Sum X Y
                                                          fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                                          ⊢ LE.le (HSub.hSub (HAdd.hAdd (f { fst := x, snd := t }) (f { fst := t, snd := …
                                                        -/
    _ ≤ f (x, z) + f (z, t) + f (t, y) - f (z, t) := by gcongr; exact candidates_triangle fA
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                  /-
                                    X : Type u
                                    Y : Type v
                                    inst✝¹ : MetricSpace X
                                    inst✝ : MetricSpace Y
                                    f : GromovHausdorff.ProdSpaceFun X Y
                                    x y z t : Sum X Y
                                    fA : Membership.mem (GromovHausdorff.candidates X Y) f
                                    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (f { fst := x, snd := z }) (f { fst := z …
                                  -/
    _ = f (x, z) + f (t, y) := by simp [sub_eq_add_neg, add_assoc]
                                  /-
                                    🎉 no goals
                                  -/
    _ ≤ maxVar X Y * dist x z + maxVar X Y * dist t y := by
      /-
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        f : GromovHausdorff.ProdSpaceFun X Y
        x y z t : Sum X Y
        fA : Membership.mem (GromovHausdorff.candidates X Y) f
        ⊢ LE.le (HAdd.hAdd (f { fst := x, snd := z }) (f { fst := t, snd := y })) (HAd …
      -/
                 /-
                   🎉 no goals
                 -/
      gcongr <;> apply candidates_dist_bound fA
                 /-
                   🎉 no goals
                 -/
    _ ≤ maxVar X Y * max (dist x z) (dist t y) + maxVar X Y * max (dist x z) (dist t y) := by
      /-
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        f : GromovHausdorff.ProdSpaceFun X Y
        x y z t : Sum X Y
        fA : Membership.mem (GromovHausdorff.candidates X Y) f
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (↑(GromovHausdorff.maxVar X Y)) (Dist.dist x z)) …
      -/
      gcongr
        /-
          case h₁.h
          X : Type u
          Y : Type v
          inst✝¹ : MetricSpace X
          inst✝ : MetricSpace Y
          f : GromovHausdorff.ProdSpaceFun X Y
          x y z t : Sum X Y
          fA : Membership.mem (GromovHausdorff.candidates X Y) f
          ⊢ LE.le (Dist.dist x z) (Max.max (Dist.dist x z) (Dist.dist t y))
        -/
      · apply le_max_left
        /-
          🎉 no goals
        -/
        /-
          case h₂.h
          X : Type u
          Y : Type v
          inst✝¹ : MetricSpace X
          inst✝ : MetricSpace Y
          f : GromovHausdorff.ProdSpaceFun X Y
          x y z t : Sum X Y
          fA : Membership.mem (GromovHausdorff.candidates X Y) f
          ⊢ LE.le (Dist.dist t y) (Max.max (Dist.dist x z) (Dist.dist t y))
        -/
      · apply le_max_right
        /-
          🎉 no goals
        -/
    _ = 2 * maxVar X Y * max (dist x z) (dist y t) := by
      /-
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        f : GromovHausdorff.ProdSpaceFun X Y
        x y z t : Sum X Y
        fA : Membership.mem (GromovHausdorff.candidates X Y) f
        ⊢ Eq (HAdd.hAdd (HMul.hMul (↑(GromovHausdorff.maxVar X Y)) (Max.max (Dist.dist …
      -/
      rw [dist_comm t y]
      /-
        X : Type u
        Y : Type v
        inst✝¹ : MetricSpace X
        inst✝ : MetricSpace Y
        f : GromovHausdorff.ProdSpaceFun X Y
        x y z t : Sum X Y
        fA : Membership.mem (GromovHausdorff.candidates X Y) f
        ⊢ Eq (HAdd.hAdd (HMul.hMul (↑(GromovHausdorff.maxVar X Y)) (Max.max (Dist.dist …
      -/
      ring
      /-
        🎉 no goals
      -/
    _ = 2 * maxVar X Y * dist (x, y) (z, t) := rfl


/-- Candidates are Lipschitz -/
private theorem candidates_lipschitz (fA : f ∈ candidates X Y) :
    LipschitzWith (2 * maxVar X Y) f := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    ⊢ LipschitzWith (HMul.hMul 2 (GromovHausdorff.maxVar X Y)) f
  -/
  apply LipschitzWith.of_dist_le_mul
  /-
    case a
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    ⊢ ∀ (x y : Prod (Sum X Y) (Sum X Y)), LE.le (Dist.dist (f x) (f y)) (HMul.hMul …
  -/
  rintro ⟨x, y⟩ ⟨z, t⟩
  /-
    case a.mk.mk
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    x y z t : Sum X Y
    ⊢ LE.le (Dist.dist (f { fst := x, snd := y }) (f { fst := z, snd := t })) (HMu …
  -/
  rw [Real.dist_eq, abs_sub_le_iff]
  /-
    case a.mk.mk
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    x y z t : Sum X Y
    ⊢ And (LE.le (HSub.hSub (f { fst := x, snd := y }) (f { fst := z, snd := t })) …
  -/
  use candidates_lipschitz_aux fA
  /-
    case right
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    x y z t : Sum X Y
    ⊢ LE.le (HSub.hSub (f { fst := z, snd := t }) (f { fst := x, snd := y })) (HMu …
  -/
  rw [dist_comm]
  /-
    case right
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    f : GromovHausdorff.ProdSpaceFun X Y
    fA : Membership.mem (GromovHausdorff.candidates X Y) f
    x y z t : Sum X Y
    ⊢ LE.le (HSub.hSub (f { fst := z, snd := t }) (f { fst := x, snd := y })) (HMu …
  -/
  exact candidates_lipschitz_aux fA
  /-
    🎉 no goals
  -/


/-- To apply Arzela-Ascoli, we need to check that the set of candidates is closed and
equicontinuous. Equicontinuity follows from the Lipschitz control, we check closedness. -/
private theorem closed_candidatesB : IsClosed (candidatesB X Y) := by
  have I1 : ∀ x y, IsClosed { f : Cb X Y | f (inl x, inl y) = dist x y } := fun x y =>
    isClosed_eq continuous_eval_const continuous_const
  have I2 : ∀ x y, IsClosed { f : Cb X Y | f (inr x, inr y) = dist x y } := fun x y =>
    isClosed_eq continuous_eval_const continuous_const
  have I3 : ∀ x y, IsClosed { f : Cb X Y | f (x, y) = f (y, x) } := fun x y =>
    isClosed_eq continuous_eval_const continuous_eval_const
  have I4 : ∀ x y z, IsClosed { f : Cb X Y | f (x, z) ≤ f (x, y) + f (y, z) } := fun x y z =>
    isClosed_le continuous_eval_const (continuous_eval_const.add continuous_eval_const)
  have I5 : ∀ x, IsClosed { f : Cb X Y | f (x, x) = 0 } := fun x =>
    isClosed_eq continuous_eval_const continuous_const
  have I6 : ∀ x y, IsClosed { f : Cb X Y | f (x, y) ≤ maxVar X Y } := fun x y =>
    isClosed_le continuous_eval_const continuous_const
  have : candidatesB X Y = (((((⋂ (x) (y), { f : Cb X Y | f (@inl X Y x, @inl X Y y) = dist x y }) ∩
      ⋂ (x) (y), { f : Cb X Y | f (@inr X Y x, @inr X Y y) = dist x y }) ∩
      ⋂ (x) (y), { f : Cb X Y | f (x, y) = f (y, x) }) ∩
      ⋂ (x) (y) (z), { f : Cb X Y | f (x, z) ≤ f (x, y) + f (y, z) }) ∩
      ⋂ x, { f : Cb X Y | f (x, x) = 0 }) ∩
      ⋂ (x) (y), { f : Cb X Y | f (x, y) ≤ maxVar X Y } := by
    ext
    simp only [candidatesB, candidates, mem_inter_iff, mem_iInter, mem_setOf_eq]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : MetricSpace X
    inst✝ : MetricSpace Y
    I1 : ∀ (x y : X), IsClosed (setOf fun f => Eq (f { fst := Sum.inl x, snd := Su …
    I2 : ∀ (x y : Y), IsClosed (setOf fun f => Eq (f { fst := Sum.inr x, snd := Su …
    I3 : ∀ (x y : Sum X Y), IsClosed (setOf fun f => Eq (f { fst := x, snd := y }) …
    I4 : ∀ (x y z : Sum X Y), IsClosed (setOf fun f => LE.le (f { fst := x, snd := …
    I5 : ∀ (x : Sum X Y), IsClosed (setOf fun f => Eq (f { fst := x, snd := x }) 0)
    I6 : ∀ (x y : Sum X Y), IsClosed (setOf fun f => LE.le (f { fst := x, snd := y …
    this : Eq (GromovHausdorff.candidatesB X Y) (Inter.inter (Inter.inter (Inter.i …
    ⊢ IsClosed (GromovHausdorff.candidatesB X Y)
  -/
  rw [this]
  repeat'
    first
      |apply IsClosed.inter _ _
      |apply isClosed_iInter _
      |apply I1 _ _|apply I2 _ _|apply I3 _ _|apply I4 _ _ _|apply I5 _|apply I6 _ _|intro x


/-- We will then choose the candidate minimizing the Hausdorff distance. Except that we are not
in a metric space setting, so we need to define our custom version of Hausdorff distance,
called `HD`, and prove its basic properties. -/
def HD (f : Cb X Y) :=
  max (⨆ x, ⨅ y, f (inl x, inr y)) (⨆ y, ⨅ x, f (inl x, inr y))

/- We will show that `HD` is continuous on `BoundedContinuousFunction`s, to deduce that its
minimum on the compact set `candidatesB` is attained. Since it is defined in terms of
infimum and supremum on `ℝ`, which is only conditionally complete, we will need all the time
to check that the defining sets are bounded below or above. This is done in the next few
technical lemmas. -/

theorem HD_below_aux1 {f : Cb X Y} (C : ℝ) {x : X} :
    BddBelow (range fun y : Y => f (inl x, inr y) + C) :=
  let ⟨cf, hcf⟩ := f.isBounded_range.bddBelow
  ⟨cf + C, forall_mem_range.2 fun _ => add_le_add_right ((fun x => hcf (mem_range_self x)) _) _⟩


private theorem HD_bound_aux1 [Nonempty Y] (f : Cb X Y) (C : ℝ) :
    BddAbove (range fun x : X => ⨅ y, f (inl x, inr y) + C) := by
  /-
    X : Type u
    Y : Type v
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Y
    f : GromovHausdorff.Cb X Y
    C : Real
    ⊢ BddAbove (Set.range fun x => iInf fun y => HAdd.hAdd (f { fst := Sum.inl x,  …
  -/
  obtain ⟨Cf, hCf⟩ := f.isBounded_range.bddAbove
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty Y
    f : GromovHausdorff.Cb X Y
    C Cf : Real
    hCf : Membership.mem (upperBounds (Set.range ⇑f)) Cf
    ⊢ BddAbove (Set.range fun x => iInf fun y => HAdd.hAdd (f { fst := Sum.inl x,  …
  -/
  refine ⟨Cf + C, forall_mem_range.2 fun x => ?_⟩
  calc
    ⨅ y, f (inl x, inr y) + C ≤ f (inl x, inr default) + C := ciInf_le (HD_below_aux1 C) default
    _ ≤ Cf + C := add_le_add ((fun x => hCf (mem_range_self x)) _) le_rfl


theorem HD_below_aux2 {f : Cb X Y} (C : ℝ) {y : Y} :
    BddBelow (range fun x : X => f (inl x, inr y) + C) :=
  let ⟨cf, hcf⟩ := f.isBounded_range.bddBelow
  ⟨cf + C, forall_mem_range.2 fun _ => add_le_add_right ((fun x => hcf (mem_range_self x)) _) _⟩


private theorem HD_bound_aux2 [Nonempty X] (f : Cb X Y) (C : ℝ) :
    BddAbove (range fun y : Y => ⨅ x, f (inl x, inr y) + C) := by
  /-
    X : Type u
    Y : Type v
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty X
    f : GromovHausdorff.Cb X Y
    C : Real
    ⊢ BddAbove (Set.range fun y => iInf fun x => HAdd.hAdd (f { fst := Sum.inl x,  …
  -/
  obtain ⟨Cf, hCf⟩ := f.isBounded_range.bddAbove
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝² : MetricSpace X
    inst✝¹ : MetricSpace Y
    inst✝ : Nonempty X
    f : GromovHausdorff.Cb X Y
    C Cf : Real
    hCf : Membership.mem (upperBounds (Set.range ⇑f)) Cf
    ⊢ BddAbove (Set.range fun y => iInf fun x => HAdd.hAdd (f { fst := Sum.inl x,  …
  -/
  refine ⟨Cf + C, forall_mem_range.2 fun y => ?_⟩
  calc
    ⨅ x, f (inl x, inr y) + C ≤ f (inl default, inr y) + C := ciInf_le (HD_below_aux2 C) default
    _ ≤ Cf + C := add_le_add ((fun x => hCf (mem_range_self x)) _) le_rfl


private theorem HD_lipschitz_aux1 (f g : Cb X Y) :
    (⨆ x, ⨅ y, f (inl x, inr y)) ≤ (⨆ x, ⨅ y, g (inl x, inr y)) + dist f g := by
  /-
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    ⊢ LE.le (iSup fun x => iInf fun y => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  obtain ⟨cg, hcg⟩ := g.isBounded_range.bddBelow
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    ⊢ LE.le (iSup fun x => iInf fun y => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  have Hcg : ∀ x, cg ≤ g x := fun x => hcg (mem_range_self x)
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    Hcg : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cg (g x)
    ⊢ LE.le (iSup fun x => iInf fun y => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  obtain ⟨cf, hcf⟩ := f.isBounded_range.bddBelow
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    Hcg : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cg (g x)
    cf : Real
    hcf : Membership.mem (lowerBounds (Set.range ⇑f)) cf
    ⊢ LE.le (iSup fun x => iInf fun y => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  have Hcf : ∀ x, cf ≤ f x := fun x => hcf (mem_range_self x)
  -- prove the inequality but with `dist f g` inside, by using inequalities comparing
  -- iSup to iSup and iInf to iInf
  have Z : (⨆ x, ⨅ y, f (inl x, inr y)) ≤ ⨆ x, ⨅ y, g (inl x, inr y) + dist f g :=
    ciSup_mono (HD_bound_aux1 _ (dist f g)) fun x =>
      ciInf_mono ⟨cf, forall_mem_range.2 fun i => Hcf _⟩ fun y => coe_le_coe_add_dist
  -- move the `dist f g` out of the infimum and the supremum, arguing that continuous monotone maps
  -- (here the addition of `dist f g`) preserve infimum and supremum
  have E1 : ∀ x, (⨅ y, g (inl x, inr y)) + dist f g = ⨅ y, g (inl x, inr y) + dist f g := by
    intro x
    refine Monotone.map_ciInf_of_continuousAt (continuousAt_id.add continuousAt_const) ?_ ?_
    · intro x y hx
      simpa
    · show BddBelow (range fun y : Y => g (inl x, inr y))
      exact ⟨cg, forall_mem_range.2 fun i => Hcg _⟩
  have E2 : (⨆ x, ⨅ y, g (inl x, inr y)) + dist f g = ⨆ x, (⨅ y, g (inl x, inr y)) + dist f g := by
    refine Monotone.map_ciSup_of_continuousAt (continuousAt_id.add continuousAt_const) ?_ ?_
    · intro x y hx
      simpa
    · simpa using HD_bound_aux1 _ 0
  -- deduce the result from the above two steps
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    Hcg : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cg (g x)
    cf : Real
    hcf : Membership.mem (lowerBounds (Set.range ⇑f)) cf
    Hcf : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cf (f x)
    Z : LE.le (iSup fun x => iInf fun y => f { fst := Sum.inl x, snd := Sum.inr y  …
    E1 : ∀ (x : X), Eq (HAdd.hAdd (iInf fun y => g { fst := Sum.inl x, snd := Sum. …
    E2 : Eq (HAdd.hAdd (iSup fun x => iInf fun y => g { fst := Sum.inl x, snd := S …
    ⊢ LE.le (iSup fun x => iInf fun y => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  simpa [E2, E1, Function.comp]
  /-
    🎉 no goals
  -/


private theorem HD_lipschitz_aux2 (f g : Cb X Y) :
    (⨆ y, ⨅ x, f (inl x, inr y)) ≤ (⨆ y, ⨅ x, g (inl x, inr y)) + dist f g := by
  /-
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    ⊢ LE.le (iSup fun y => iInf fun x => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  obtain ⟨cg, hcg⟩ := g.isBounded_range.bddBelow
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    ⊢ LE.le (iSup fun y => iInf fun x => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  have Hcg : ∀ x, cg ≤ g x := fun x => hcg (mem_range_self x)
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    Hcg : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cg (g x)
    ⊢ LE.le (iSup fun y => iInf fun x => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  obtain ⟨cf, hcf⟩ := f.isBounded_range.bddBelow
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    Hcg : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cg (g x)
    cf : Real
    hcf : Membership.mem (lowerBounds (Set.range ⇑f)) cf
    ⊢ LE.le (iSup fun y => iInf fun x => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  have Hcf : ∀ x, cf ≤ f x := fun x => hcf (mem_range_self x)
  -- prove the inequality but with `dist f g` inside, by using inequalities comparing
  -- iSup to iSup and iInf to iInf
  have Z : (⨆ y, ⨅ x, f (inl x, inr y)) ≤ ⨆ y, ⨅ x, g (inl x, inr y) + dist f g :=
    ciSup_mono (HD_bound_aux2 _ (dist f g)) fun y =>
      ciInf_mono ⟨cf, forall_mem_range.2 fun i => Hcf _⟩ fun y => coe_le_coe_add_dist
  -- move the `dist f g` out of the infimum and the supremum, arguing that continuous monotone maps
  -- (here the addition of `dist f g`) preserve infimum and supremum
  have E1 : ∀ y, (⨅ x, g (inl x, inr y)) + dist f g = ⨅ x, g (inl x, inr y) + dist f g := by
    intro y
    refine Monotone.map_ciInf_of_continuousAt (continuousAt_id.add continuousAt_const) ?_ ?_
    · intro x y hx
      simpa
    · show BddBelow (range fun x : X => g (inl x, inr y))
      exact ⟨cg, forall_mem_range.2 fun i => Hcg _⟩
  have E2 : (⨆ y, ⨅ x, g (inl x, inr y)) + dist f g = ⨆ y, (⨅ x, g (inl x, inr y)) + dist f g := by
    refine Monotone.map_ciSup_of_continuousAt (continuousAt_id.add continuousAt_const) ?_ ?_
    · intro x y hx
      simpa
    · simpa using HD_bound_aux2 _ 0
  -- deduce the result from the above two steps
  /-
    case intro.intro
    X : Type u
    Y : Type v
    inst✝³ : MetricSpace X
    inst✝² : MetricSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    f g : GromovHausdorff.Cb X Y
    cg : Real
    hcg : Membership.mem (lowerBounds (Set.range ⇑g)) cg
    Hcg : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cg (g x)
    cf : Real
    hcf : Membership.mem (lowerBounds (Set.range ⇑f)) cf
    Hcf : ∀ (x : Prod (Sum X Y) (Sum X Y)), LE.le cf (f x)
    Z : LE.le (iSup fun y => iInf fun x => f { fst := Sum.inl x, snd := Sum.inr y  …
    E1 : ∀ (y : Y), Eq (HAdd.hAdd (iInf fun x => g { fst := Sum.inl x, snd := Sum. …
    E2 : Eq (HAdd.hAdd (iSup fun y => iInf fun x => g { fst := Sum.inl x, snd := S …
    ⊢ LE.le (iSup fun y => iInf fun x => f { fst := Sum.inl x, snd := Sum.inr y }) …
  -/
  simpa [E2, E1]
  /-
    🎉 no goals
  -/


private theorem HD_lipschitz_aux3 (f g : Cb X Y) :
    HD f ≤ HD g + dist f g :=
  max_le (le_trans (HD_lipschitz_aux1 f g) (add_le_add_right (le_max_left _ _) _))
    (le_trans (HD_lipschitz_aux2 f g) (add_le_add_right (le_max_right _ _) _))


/-- Conclude that `HD`, being Lipschitz, is continuous -/
private theorem HD_continuous : Continuous (HD : Cb X Y → ℝ) :=
  LipschitzWith.continuous (LipschitzWith.of_le_add HD_lipschitz_aux3)


/-- Compactness of candidates (in `BoundedContinuousFunction`s) follows. -/
private theorem isCompact_candidatesB : IsCompact (candidatesB X Y) := by
  refine arzela_ascoli₂
      (Icc 0 (maxVar X Y) : Set ℝ) isCompact_Icc (candidatesB X Y) closed_candidatesB ?_ ?_
    /-
      case refine_1
      X : Type u
      Y : Type v
      inst✝³ : MetricSpace X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace X
      inst✝ : CompactSpace Y
      ⊢ ∀ (f : BoundedContinuousFunction (Prod (Sum X Y) (Sum X Y)) Real) (x : Prod  …
    -/
  · rintro f ⟨x1, x2⟩ hf
    /-
      case refine_1.mk
      X : Type u
      Y : Type v
      inst✝³ : MetricSpace X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace X
      inst✝ : CompactSpace Y
      f : BoundedContinuousFunction (Prod (Sum X Y) (Sum X Y)) Real
      x1 x2 : Sum X Y
      hf : Membership.mem (GromovHausdorff.candidatesB X Y) f
      ⊢ Membership.mem (Set.Icc 0 ↑(GromovHausdorff.maxVar X Y)) (f { fst := x1, snd …
    -/
    simp only [Set.mem_Icc]
    /-
      case refine_1.mk
      X : Type u
      Y : Type v
      inst✝³ : MetricSpace X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace X
      inst✝ : CompactSpace Y
      f : BoundedContinuousFunction (Prod (Sum X Y) (Sum X Y)) Real
      x1 x2 : Sum X Y
      hf : Membership.mem (GromovHausdorff.candidatesB X Y) f
      ⊢ And (LE.le 0 (f { fst := x1, snd := x2 })) (LE.le (f { fst := x1, snd := x2  …
    -/
    exact ⟨candidates_nonneg hf, candidates_le_maxVar hf⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      Y : Type v
      inst✝³ : MetricSpace X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace X
      inst✝ : CompactSpace Y
      ⊢ Equicontinuous fun x => ⇑↑x
    -/
  · refine equicontinuous_of_continuity_modulus (fun t => 2 * maxVar X Y * t) ?_ _ ?_
    · have : Tendsto (fun t : ℝ => 2 * (maxVar X Y : ℝ) * t) (𝓝 0) (𝓝 (2 * maxVar X Y * 0)) :=
        tendsto_const_nhds.mul tendsto_id
      /-
        case refine_2.refine_1
        X : Type u
        Y : Type v
        inst✝³ : MetricSpace X
        inst✝² : MetricSpace Y
        inst✝¹ : CompactSpace X
        inst✝ : CompactSpace Y
        this : Filter.Tendsto (fun t => HMul.hMul (HMul.hMul 2 ↑(GromovHausdorff.maxVa …
        ⊢ Filter.Tendsto (fun t => HMul.hMul (HMul.hMul 2 ↑(GromovHausdorff.maxVar X Y …
      -/
      simpa using this
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        X : Type u
        Y : Type v
        inst✝³ : MetricSpace X
        inst✝² : MetricSpace Y
        inst✝¹ : CompactSpace X
        inst✝ : CompactSpace Y
        ⊢ ∀ (x y : Prod (Sum X Y) (Sum X Y)) (i : ↑(GromovHausdorff.candidatesB X Y)), …
      -/
    · rintro x y ⟨f, hf⟩
      /-
        case refine_2.refine_2.mk
        X : Type u
        Y : Type v
        inst✝³ : MetricSpace X
        inst✝² : MetricSpace Y
        inst✝¹ : CompactSpace X
        inst✝ : CompactSpace Y
        x y : Prod (Sum X Y) (Sum X Y)
        f : BoundedContinuousFunction (Prod (Sum X Y) (Sum X Y)) Real
        hf : Membership.mem (GromovHausdorff.candidatesB X Y) f
        ⊢ LE.le (Dist.dist (↑⟨f, hf⟩ x) (↑⟨f, hf⟩ y)) ((fun t => HMul.hMul (HMul.hMul  …
      -/
      exact (candidates_lipschitz hf).dist_le_mul _ _
      /-
        🎉 no goals
      -/


/-- candidates give rise to elements of `BoundedContinuousFunction`s -/
def candidatesBOfCandidates (f : ProdSpaceFun X Y) (fA : f ∈ candidates X Y) : Cb X Y :=
  BoundedContinuousFunction.mkOfCompact ⟨f, (candidates_lipschitz fA).continuous⟩


theorem candidatesBOfCandidates_mem (f : ProdSpaceFun X Y) (fA : f ∈ candidates X Y) :
    candidatesBOfCandidates f fA ∈ candidatesB X Y :=
  fA


/-- The distance on `X ⊕ Y` is a candidate -/
private theorem dist_mem_candidates :
    (fun p : (X ⊕ Y) × (X ⊕ Y) => dist p.1 p.2) ∈ candidates X Y := by
  simp_rw [candidates, Set.mem_setOf_eq, dist_comm, dist_triangle, dist_self, maxVar_bound,
    forall_const, and_true]
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : MetricSpace Y
    inst✝³ : CompactSpace X
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    ⊢ And (∀ (x y : X), Eq (Dist.dist (Sum.inl x) (Sum.inl y)) (Dist.dist x y)) (∀ …
  -/
  exact ⟨fun x y => rfl, fun x y => rfl⟩
  /-
    🎉 no goals
  -/


/-- The distance on `X ⊕ Y` as a candidate -/
def candidatesBDist (X : Type u) (Y : Type v) [MetricSpace X] [CompactSpace X] [Nonempty X]
    [MetricSpace Y] [CompactSpace Y] [Nonempty Y] : Cb X Y :=
  candidatesBOfCandidates _ dist_mem_candidates


theorem candidatesBDist_mem_candidatesB :
    candidatesBDist X Y ∈ candidatesB X Y :=
  candidatesBOfCandidates_mem _ _


private theorem candidatesB_nonempty : (candidatesB X Y).Nonempty :=
  ⟨_, candidatesBDist_mem_candidatesB⟩


/-- Explicit bound on `HD (dist)`. This means that when looking for minimizers it will
be sufficient to look for functions with `HD(f)` bounded by this bound. -/
theorem HD_candidatesBDist_le :
    HD (candidatesBDist X Y) ≤ diam (univ : Set X) + 1 + diam (univ : Set Y) := by
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : MetricSpace Y
    inst✝³ : CompactSpace X
    inst✝² : CompactSpace Y
    inst✝¹ : Nonempty X
    inst✝ : Nonempty Y
    ⊢ LE.le (GromovHausdorff.HD (GromovHausdorff.candidatesBDist X Y)) (HAdd.hAdd  …
  -/
  refine max_le (ciSup_le fun x => ?_) (ciSup_le fun y => ?_)
  · have A : ⨅ y, candidatesBDist X Y (inl x, inr y) ≤ candidatesBDist X Y (inl x, inr default) :=
      ciInf_le (by simpa using HD_below_aux1 0) default
    have B : dist (inl x) (inr default) ≤ diam (univ : Set X) + 1 + diam (univ : Set Y) :=
      calc
        dist (inl x) (inr (default : Y)) = dist x (default : X) + 1 + dist default default := rfl
        _ ≤ diam (univ : Set X) + 1 + diam (univ : Set Y) := by
          gcongr <;>
            exact dist_le_diam_of_mem isBounded_of_compactSpace (mem_univ _) (mem_univ _)
    /-
      case refine_1
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : MetricSpace Y
      inst✝³ : CompactSpace X
      inst✝² : CompactSpace Y
      inst✝¹ : Nonempty X
      inst✝ : Nonempty Y
      x : X
      A : LE.le (iInf fun y => (GromovHausdorff.candidatesBDist X Y) { fst := Sum.in …
      B : LE.le (Dist.dist (Sum.inl x) (Sum.inr Inhabited.default)) (HAdd.hAdd (HAdd …
      ⊢ LE.le (iInf fun y => (GromovHausdorff.candidatesBDist X Y) { fst := Sum.inl  …
    -/
    exact le_trans A B
    /-
      🎉 no goals
    -/
  · have A : ⨅ x, candidatesBDist X Y (inl x, inr y) ≤ candidatesBDist X Y (inl default, inr y) :=
      ciInf_le (by simpa using HD_below_aux2 0) default
    have B : dist (inl default) (inr y) ≤ diam (univ : Set X) + 1 + diam (univ : Set Y) :=
      calc
        dist (inl (default : X)) (inr y) = dist default default + 1 + dist default y := rfl
        _ ≤ diam (univ : Set X) + 1 + diam (univ : Set Y) := by
          gcongr <;>
            exact dist_le_diam_of_mem isBounded_of_compactSpace (mem_univ _) (mem_univ _)
    /-
      case refine_2
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : MetricSpace Y
      inst✝³ : CompactSpace X
      inst✝² : CompactSpace Y
      inst✝¹ : Nonempty X
      inst✝ : Nonempty Y
      y : Y
      A : LE.le (iInf fun x => (GromovHausdorff.candidatesBDist X Y) { fst := Sum.in …
      B : LE.le (Dist.dist (Sum.inl Inhabited.default) (Sum.inr y)) (HAdd.hAdd (HAdd …
      ⊢ LE.le (iInf fun x => (GromovHausdorff.candidatesBDist X Y) { fst := Sum.inl  …
    -/
    exact le_trans A B
    /-
      🎉 no goals
    -/


private theorem exists_minimizer : ∃ f ∈ candidatesB X Y, ∀ g ∈ candidatesB X Y, HD f ≤ HD g :=
  isCompact_candidatesB.exists_isMinOn candidatesB_nonempty HD_continuous.continuousOn


private def optimalGHDist : Cb X Y :=
  Classical.choose (exists_minimizer X Y)


private theorem optimalGHDist_mem_candidatesB : optimalGHDist X Y ∈ candidatesB X Y := by
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    ⊢ Membership.mem (GromovHausdorff.candidatesB X Y) (GromovHausdorff.optimalGHD …
  -/
  cases Classical.choose_spec (exists_minimizer X Y)
  /-
    case intro
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    left✝ : Membership.mem (GromovHausdorff.candidatesB X Y) (Classical.choose ⋯)
    right✝ : ∀ (g : GromovHausdorff.Cb X Y), Membership.mem (GromovHausdorff.candi …
    ⊢ Membership.mem (GromovHausdorff.candidatesB X Y) (GromovHausdorff.optimalGHD …
  -/
  assumption
  /-
    🎉 no goals
  -/


private theorem HD_optimalGHDist_le (g : Cb X Y) (hg : g ∈ candidatesB X Y) :
    HD (optimalGHDist X Y) ≤ HD g :=
  let ⟨_, Z2⟩ := Classical.choose_spec (exists_minimizer X Y)
  Z2 g hg


/-- With the optimal candidate, construct a premetric space structure on `X ⊕ Y`, on which the
predistance is given by the candidate. Then, we will identify points at `0` predistance
to obtain a genuine metric space. -/
def premetricOptimalGHDist : PseudoMetricSpace (X ⊕ Y) where
  dist p q := optimalGHDist X Y (p, q)
  dist_self _ := candidates_refl (optimalGHDist_mem_candidatesB X Y)
  dist_comm _ _ := candidates_symm (optimalGHDist_mem_candidatesB X Y)
  dist_triangle _ _ _ := candidates_triangle (optimalGHDist_mem_candidatesB X Y)
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): added proof for `edist_dist`
  edist_dist x y := by
    /-
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      x y : Sum X Y
      ⊢ Eq ((fun x y => ↑⟨(GromovHausdorff.optimalGHDist X Y) { fst := x, snd := y } …
    -/
    simp only
    /-
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      x y : Sum X Y
      ⊢ Eq (↑⟨(GromovHausdorff.optimalGHDist X Y) { fst := x, snd := y }, ⋯⟩) (ENNRe …
    -/
    congr
    /-
      case e_a.e_val
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      x y : Sum X Y
      ⊢ Eq ((GromovHausdorff.optimalGHDist X Y) { fst := x, snd := y }) (Max.max ((G …
    -/
    simp only [left_eq_sup]
    /-
      case e_a.e_val
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      x y : Sum X Y
      ⊢ LE.le 0 ((GromovHausdorff.optimalGHDist X Y) { fst := x, snd := y })
    -/
    exact candidates_nonneg (optimalGHDist_mem_candidatesB X Y)
    /-
      🎉 no goals
    -/


/-- A metric space which realizes the optimal coupling between `X` and `Y` -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): This linter does not exist yet.
def OptimalGHCoupling : Type _ :=
  @SeparationQuotient (X ⊕ Y) (premetricOptimalGHDist X Y).toUniformSpace.toTopologicalSpace


instance : MetricSpace (OptimalGHCoupling X Y) := by
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    ⊢ MetricSpace (GromovHausdorff.OptimalGHCoupling X Y)
  -/
  unfold OptimalGHCoupling
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    ⊢ MetricSpace (SeparationQuotient (Sum X Y))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Injection of `X` in the optimal coupling between `X` and `Y` -/
def optimalGHInjl (x : X) : OptimalGHCoupling X Y :=
  Quotient.mk'' (inl x)


/-- The injection of `X` in the optimal coupling between `X` and `Y` is an isometry. -/
theorem isometry_optimalGHInjl : Isometry (optimalGHInjl X Y) :=
  Isometry.of_dist_eq fun _ _ => candidates_dist_inl (optimalGHDist_mem_candidatesB X Y) _ _


/-- Injection of `Y` in the optimal coupling between `X` and `Y` -/
def optimalGHInjr (y : Y) : OptimalGHCoupling X Y :=
  Quotient.mk'' (inr y)


/-- The injection of `Y` in the optimal coupling between `X` and `Y` is an isometry. -/
theorem isometry_optimalGHInjr : Isometry (optimalGHInjr X Y) :=
  Isometry.of_dist_eq fun _ _ => candidates_dist_inr (optimalGHDist_mem_candidatesB X Y) _ _


/-- The optimal coupling between two compact spaces `X` and `Y` is still a compact space -/
instance compactSpace_optimalGHCoupling : CompactSpace (OptimalGHCoupling X Y) := ⟨by
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    ⊢ IsCompact Set.univ
  -/
  rw [← range_quotient_mk']
  exact isCompact_range (continuous_sum_dom.2
    ⟨(isometry_optimalGHInjl X Y).continuous, (isometry_optimalGHInjr X Y).continuous⟩)⟩


/-- For any candidate `f`, `HD(f)` is larger than or equal to the Hausdorff distance in the
optimal coupling. This follows from the fact that `HD` of the optimal candidate is exactly
the Hausdorff distance in the optimal coupling, although we only prove here the inequality
we need. -/
theorem hausdorffDist_optimal_le_HD {f} (h : f ∈ candidatesB X Y) :
    hausdorffDist (range (optimalGHInjl X Y)) (range (optimalGHInjr X Y)) ≤ HD f := by
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    f : GromovHausdorff.Cb X Y
    h : Membership.mem (GromovHausdorff.candidatesB X Y) f
    ⊢ LE.le (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y))  …
  -/
  refine le_trans (le_of_forall_le_of_dense fun r hr => ?_) (HD_optimalGHDist_le X Y f h)
  have A : ∀ x ∈ range (optimalGHInjl X Y), ∃ y ∈ range (optimalGHInjr X Y), dist x y ≤ r := by
    rintro _ ⟨z, rfl⟩
    have I1 : (⨆ x, ⨅ y, optimalGHDist X Y (inl x, inr y)) < r :=
      lt_of_le_of_lt (le_max_left _ _) hr
    have I2 :
        ⨅ y, optimalGHDist X Y (inl z, inr y) ≤ ⨆ x, ⨅ y, optimalGHDist X Y (inl x, inr y) :=
      le_csSup (by simpa using HD_bound_aux1 _ 0) (mem_range_self _)
    have I : ⨅ y, optimalGHDist X Y (inl z, inr y) < r := lt_of_le_of_lt I2 I1
    rcases exists_lt_of_csInf_lt (range_nonempty _) I with ⟨r', ⟨z', rfl⟩, hr'⟩
    exact ⟨optimalGHInjr X Y z', mem_range_self _, le_of_lt hr'⟩
  /-
    X : Type u
    Y : Type v
    inst✝⁵ : MetricSpace X
    inst✝⁴ : CompactSpace X
    inst✝³ : Nonempty X
    inst✝² : MetricSpace Y
    inst✝¹ : CompactSpace Y
    inst✝ : Nonempty Y
    f : GromovHausdorff.Cb X Y
    h : Membership.mem (GromovHausdorff.candidatesB X Y) f
    r : Real
    hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
    A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
    ⊢ LE.le (Metric.hausdorffDist (Set.range (GromovHausdorff.optimalGHInjl X Y))  …
  -/
  refine hausdorffDist_le_of_mem_dist ?_ A ?_
    /-
      case refine_1
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      ⊢ LE.le 0 r
    -/
  · inhabit X
    /-
      case refine_1
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      inhabited_h : Inhabited X
      ⊢ LE.le 0 r
    -/
    rcases A _ (mem_range_self default) with ⟨y, -, hy⟩
    /-
      case refine_1.intro.intro
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      inhabited_h : Inhabited X
      y : GromovHausdorff.OptimalGHCoupling X Y
      hy : LE.le (Dist.dist (GromovHausdorff.optimalGHInjl X Y Inhabited.default) y) r
      ⊢ LE.le 0 r
    -/
    exact le_trans dist_nonneg hy
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      ⊢ ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range (Gr …
    -/
  · rintro _ ⟨z, rfl⟩
    have I1 : (⨆ y, ⨅ x, optimalGHDist X Y (inl x, inr y)) < r :=
      lt_of_le_of_lt (le_max_right _ _) hr
    have I2 :
        ⨅ x, optimalGHDist X Y (inl x, inr z) ≤ ⨆ y, ⨅ x, optimalGHDist X Y (inl x, inr y) :=
      le_csSup (by simpa using HD_bound_aux2 _ 0) (mem_range_self _)
    /-
      case refine_2.intro
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      z : Y
      I1 : LT.lt (iSup fun y => iInf fun x => (GromovHausdorff.optimalGHDist X Y) {  …
      I2 : LE.le (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl …
      ⊢ Exists fun y => And (Membership.mem (Set.range (GromovHausdorff.optimalGHInj …
    -/
    have I : ⨅ x, optimalGHDist X Y (inl x, inr z) < r := lt_of_le_of_lt I2 I1
    /-
      case refine_2.intro
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      z : Y
      I1 : LT.lt (iSup fun y => iInf fun x => (GromovHausdorff.optimalGHDist X Y) {  …
      I2 : LE.le (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl …
      I : LT.lt (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl  …
      ⊢ Exists fun y => And (Membership.mem (Set.range (GromovHausdorff.optimalGHInj …
    -/
    rcases exists_lt_of_csInf_lt (range_nonempty _) I with ⟨r', ⟨z', rfl⟩, hr'⟩
    /-
      case refine_2.intro.intro.intro.intro
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      z : Y
      I1 : LT.lt (iSup fun y => iInf fun x => (GromovHausdorff.optimalGHDist X Y) {  …
      I2 : LE.le (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl …
      I : LT.lt (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl  …
      z' : X
      hr' : LT.lt ((fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl x, …
      ⊢ Exists fun y => And (Membership.mem (Set.range (GromovHausdorff.optimalGHInj …
    -/
    refine ⟨optimalGHInjl X Y z', mem_range_self _, le_of_lt ?_⟩
    /-
      case refine_2.intro.intro.intro.intro
      X : Type u
      Y : Type v
      inst✝⁵ : MetricSpace X
      inst✝⁴ : CompactSpace X
      inst✝³ : Nonempty X
      inst✝² : MetricSpace Y
      inst✝¹ : CompactSpace Y
      inst✝ : Nonempty Y
      f : GromovHausdorff.Cb X Y
      h : Membership.mem (GromovHausdorff.candidatesB X Y) f
      r : Real
      hr : LT.lt (GromovHausdorff.HD (GromovHausdorff.optimalGHDist X Y)) r
      A : ∀ (x : GromovHausdorff.OptimalGHCoupling X Y), Membership.mem (Set.range ( …
      z : Y
      I1 : LT.lt (iSup fun y => iInf fun x => (GromovHausdorff.optimalGHDist X Y) {  …
      I2 : LE.le (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl …
      I : LT.lt (iInf fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl  …
      z' : X
      hr' : LT.lt ((fun x => (GromovHausdorff.optimalGHDist X Y) { fst := Sum.inl x, …
      ⊢ LT.lt (Dist.dist (GromovHausdorff.optimalGHInjr X Y z) (GromovHausdorff.opti …
    -/
    rwa [dist_comm]
    /-
      🎉 no goals
    -/


