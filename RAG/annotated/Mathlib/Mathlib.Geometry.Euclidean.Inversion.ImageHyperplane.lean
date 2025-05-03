/-- The inversion with center `c` and radius `R` maps a sphere passing through the center to a
hyperplane. -/
theorem inversion_mem_perpBisector_inversion_iff (hR : R ≠ 0) (hx : x ≠ c) (hy : y ≠ c) :
    inversion c R x ∈ perpBisector c (inversion c R y) ↔ dist x y = dist y c := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hR : Ne R 0
    hx : Ne x c
    hy : Ne y c
    ⊢ Iff (Membership.mem (AffineSubspace.perpBisector c (EuclideanGeometry.invers …
  -/
  rw [mem_perpBisector_iff_dist_eq, dist_inversion_inversion hx hy, dist_inversion_center]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hR : Ne R 0
    hx : Ne x c
    hy : Ne y c
    ⊢ Iff (Eq (HDiv.hDiv (HPow.hPow R 2) (Dist.dist x c)) (HMul.hMul (HDiv.hDiv (H …
  -/
  have hx' := dist_ne_zero.2 hx
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hR : Ne R 0
    hx : Ne x c
    hy : Ne y c
    hx' : Ne (Dist.dist x c) 0
    ⊢ Iff (Eq (HDiv.hDiv (HPow.hPow R 2) (Dist.dist x c)) (HMul.hMul (HDiv.hDiv (H …
  -/
  have hy' := dist_ne_zero.2 hy
  -- takes 300ms, but the "equivalent" simp call fails -> hard to speed up
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hR : Ne R 0
    hx : Ne x c
    hy : Ne y c
    hx' : Ne (Dist.dist x c) 0
    hy' : Ne (Dist.dist y c) 0
    ⊢ Iff (Eq (HDiv.hDiv (HPow.hPow R 2) (Dist.dist x c)) (HMul.hMul (HDiv.hDiv (H …
  -/
  field_simp [mul_assoc, mul_comm, hx, hx.symm, eq_comm]
  /-
    🎉 no goals
  -/


/-- The inversion with center `c` and radius `R` maps a sphere passing through the center to a
hyperplane. -/
theorem inversion_mem_perpBisector_inversion_iff' (hR : R ≠ 0) (hy : y ≠ c) :
    inversion c R x ∈ perpBisector c (inversion c R y) ↔ dist x y = dist y c ∧ x ≠ c := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c x y : P
    R : Real
    hR : Ne R 0
    hy : Ne y c
    ⊢ Iff (Membership.mem (AffineSubspace.perpBisector c (EuclideanGeometry.invers …
  -/
  rcases eq_or_ne x c with rfl | hx
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      x y : P
      R : Real
      hR : Ne R 0
      hy : Ne y x
      ⊢ Iff (Membership.mem (AffineSubspace.perpBisector x (EuclideanGeometry.invers …
    -/
  · simp [*]
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      P : Type u_2
      inst✝³ : NormedAddCommGroup V
      inst✝² : InnerProductSpace Real V
      inst✝¹ : MetricSpace P
      inst✝ : NormedAddTorsor V P
      c x y : P
      R : Real
      hR : Ne R 0
      hy : Ne y c
      hx : Ne x c
      ⊢ Iff (Membership.mem (AffineSubspace.perpBisector c (EuclideanGeometry.invers …
    -/
  · simp [inversion_mem_perpBisector_inversion_iff hR hx hy, hx]
    /-
      🎉 no goals
    -/


theorem preimage_inversion_perpBisector_inversion (hR : R ≠ 0) (hy : y ≠ c) :
    inversion c R ⁻¹' perpBisector c (inversion c R y) = sphere y (dist y c) \ {c} :=
  Set.ext fun _ ↦ inversion_mem_perpBisector_inversion_iff' hR hy


theorem preimage_inversion_perpBisector (hR : R ≠ 0) (hy : y ≠ c) :
    inversion c R ⁻¹' perpBisector c y = sphere (inversion c R y) (R ^ 2 / dist y c) \ {c} := by
  rw [← dist_inversion_center, ← preimage_inversion_perpBisector_inversion hR,
                             /-
                               case hR
                               V : Type u_1
                               P : Type u_2
                               inst✝³ : NormedAddCommGroup V
                               inst✝² : InnerProductSpace Real V
                               inst✝¹ : MetricSpace P
                               inst✝ : NormedAddTorsor V P
                               c y : P
                               R : Real
                               hR : Ne R 0
                               hy : Ne y c
                               ⊢ Ne R 0
                             -/
                             /-
                               🎉 no goals
                             -/
    inversion_inversion] <;> simp [*]
                             /-
                               🎉 no goals
                             -/


theorem image_inversion_perpBisector (hR : R ≠ 0) (hy : y ≠ c) :
    inversion c R '' perpBisector c y = sphere (inversion c R y) (R ^ 2 / dist y c) \ {c} := by
  rw [image_eq_preimage_of_inverse (inversion_involutive _ hR) (inversion_involutive _ hR),
    preimage_inversion_perpBisector hR hy]


theorem preimage_inversion_sphere_dist_center (hR : R ≠ 0) (hy : y ≠ c) :
    inversion c R ⁻¹' sphere y (dist y c) =
      insert c (perpBisector c (inversion c R y) : Set P) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c y : P
    R : Real
    hR : Ne R 0
    hy : Ne y c
    ⊢ Eq (Set.preimage (EuclideanGeometry.inversion c R) (Metric.sphere y (Dist.di …
  -/
  ext x
  /-
    case h
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c y : P
    R : Real
    hR : Ne R 0
    hy : Ne y c
    x : P
    ⊢ Iff (Membership.mem (Set.preimage (EuclideanGeometry.inversion c R) (Metric. …
  -/
  rcases eq_or_ne x c with rfl | hx; · simp [dist_comm]
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case h.inr
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    c y : P
    R : Real
    hR : Ne R 0
    hy : Ne y c
    x : P
    hx : Ne x c
    ⊢ Iff (Membership.mem (Set.preimage (EuclideanGeometry.inversion c R) (Metric. …
  -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  rw [mem_preimage, mem_sphere, ← inversion_mem_perpBisector_inversion_iff hR] <;> simp [*]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem image_inversion_sphere_dist_center (hR : R ≠ 0) (hy : y ≠ c) :
    inversion c R '' sphere y (dist y c) = insert c (perpBisector c (inversion c R y) : Set P) := by
  rw [image_eq_preimage_of_inverse (inversion_involutive _ hR) (inversion_involutive _ hR),
    preimage_inversion_sphere_dist_center hR hy]


/-- Inversion sends an affine subspace passing through the center to itself. -/
theorem mapsTo_inversion_affineSubspace_of_mem {p : AffineSubspace ℝ P} (hp : c ∈ p) :
    MapsTo (inversion c R) p p := fun _ ↦ AffineMap.lineMap_mem _ hp


/-- Inversion sends an affine subspace passing through the center to itself. -/
theorem image_inversion_affineSubspace_of_mem {p : AffineSubspace ℝ P} (hR : R ≠ 0) (hp : c ∈ p) :
    inversion c R '' p = p :=
  (mapsTo_inversion_affineSubspace_of_mem hp).image_subset.antisymm fun x hx ↦
    ⟨inversion c R x, mapsTo_inversion_affineSubspace_of_mem hp hx, inversion_inversion _ hR _⟩


