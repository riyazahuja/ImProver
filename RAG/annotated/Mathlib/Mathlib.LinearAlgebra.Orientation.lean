/-- An orientation of a module, intended to be used when `ι` is a `Fintype` with the same
cardinality as a basis. -/
abbrev Orientation := Module.Ray R (M [⋀^ι]→ₗ[R] R)


/-- A type class fixing an orientation of a module. -/
class Module.Oriented where
  /-- Fix a positive orientation. -/
  positiveOrientation : Orientation R M ι


/-- An equivalence between modules implies an equivalence between orientations. -/
def Orientation.map (e : M ≃ₗ[R] N) : Orientation R M ι ≃ Orientation R N ι :=
  Module.Ray.map <| AlternatingMap.domLCongr R R ι R e


@[simp]
theorem Orientation.map_apply (e : M ≃ₗ[R] N) (v : M [⋀^ι]→ₗ[R] R) (hv : v ≠ 0) :
    Orientation.map ι e (rayOfNeZero _ v hv) =
      rayOfNeZero _ (v.compLinearMap e.symm) (mt (v.compLinearEquiv_eq_zero_iff e.symm).mp hv) :=
  rfl


@[simp]
theorem Orientation.map_refl : (Orientation.map ι <| LinearEquiv.refl R M) = Equiv.refl _ := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_4
    ⊢ Eq (Orientation.map ι (LinearEquiv.refl R M)) (Equiv.refl (Orientation R M ι))
  -/
  rw [Orientation.map, AlternatingMap.domLCongr_refl, Module.Ray.map_refl]
  /-
    🎉 no goals
  -/


@[simp]
theorem Orientation.map_symm (e : M ≃ₗ[R] N) :
    (Orientation.map ι e).symm = Orientation.map ι e.symm := rfl


/-- An equivalence between indices implies an equivalence between orientations. -/
def Orientation.reindex (e : ι ≃ ι') : Orientation R M ι ≃ Orientation R M ι' :=
  Module.Ray.map <| AlternatingMap.domDomCongrₗ R e


@[simp]
theorem Orientation.reindex_apply (e : ι ≃ ι') (v : M [⋀^ι]→ₗ[R] R) (hv : v ≠ 0) :
    Orientation.reindex R M e (rayOfNeZero _ v hv) =
      rayOfNeZero _ (v.domDomCongr e) (mt (v.domDomCongr_eq_zero_iff e).mp hv) :=
  rfl


@[simp]
theorem Orientation.reindex_refl : (Orientation.reindex R M <| Equiv.refl ι) = Equiv.refl _ := by
  /-
    R : Type u_1
    inst✝² : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Type u_4
    ⊢ Eq (Orientation.reindex R M (Equiv.refl ι)) (Equiv.refl (Orientation R M ι))
  -/
  rw [Orientation.reindex, AlternatingMap.domDomCongrₗ_refl, Module.Ray.map_refl]
  /-
    🎉 no goals
  -/


@[simp]
theorem Orientation.reindex_symm (e : ι ≃ ι') :
    (Orientation.reindex R M e).symm = Orientation.reindex R M e.symm :=
  rfl


/-- A module is canonically oriented with respect to an empty index type. -/
instance (priority := 100) IsEmpty.oriented [IsEmpty ι] : Module.Oriented R M ι where
  positiveOrientation :=
    rayOfNeZero R (AlternatingMap.constLinearEquivOfIsEmpty 1) <|
                                                                /-
                                                                  R : Type u_1
                                                                  inst✝⁵ : StrictOrderedCommSemiring R
                                                                  M : Type u_2
                                                                  inst✝⁴ : AddCommMonoid M
                                                                  inst✝³ : Module R M
                                                                  N : Type u_3
                                                                  inst✝² : AddCommMonoid N
                                                                  inst✝¹ : Module R N
                                                                  ι : Type u_4
                                                                  ι' : Type u_5
                                                                  inst✝ : IsEmpty ι
                                                                  ⊢ Ne 1 0
                                                                -/
      AlternatingMap.constLinearEquivOfIsEmpty.injective.ne (by exact one_ne_zero)
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem Orientation.map_positiveOrientation_of_isEmpty [IsEmpty ι] (f : M ≃ₗ[R] N) :
    Orientation.map ι f positiveOrientation = positiveOrientation := rfl


@[simp]
theorem Orientation.map_of_isEmpty [IsEmpty ι] (x : Orientation R M ι) (f : M ≃ₗ[R] M) :
    Orientation.map ι f x = x := by
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    ⊢ Eq ((Orientation.map ι f) x) x
  -/
  induction' x using Module.Ray.ind with g hg
  /-
    case h
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    f : LinearEquiv (RingHom.id R) M M
    g : AlternatingMap R M R ι
    hg : Ne g 0
    ⊢ Eq ((Orientation.map ι f) (rayOfNeZero R g hg)) (rayOfNeZero R g hg)
  -/
  rw [Orientation.map_apply]
  /-
    case h
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    f : LinearEquiv (RingHom.id R) M M
    g : AlternatingMap R M R ι
    hg : Ne g 0
    ⊢ Eq (rayOfNeZero R (g.compLinearMap ↑f.symm) ⋯) (rayOfNeZero R g hg)
  -/
  congr
  /-
    case h.e_v
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    f : LinearEquiv (RingHom.id R) M M
    g : AlternatingMap R M R ι
    hg : Ne g 0
    ⊢ Eq (g.compLinearMap ↑f.symm) g
  -/
  ext i
  /-
    case h.e_v.H
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    f : LinearEquiv (RingHom.id R) M M
    g : AlternatingMap R M R ι
    hg : Ne g 0
    i : ι → M
    ⊢ Eq ((g.compLinearMap ↑f.symm) i) (g i)
  -/
  rw [AlternatingMap.compLinearMap_apply]
  /-
    case h.e_v.H
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    f : LinearEquiv (RingHom.id R) M M
    g : AlternatingMap R M R ι
    hg : Ne g 0
    i : ι → M
    ⊢ Eq (g fun i_1 => ↑f.symm (i i_1)) (g i)
  -/
  congr
  /-
    case h.e_v.H.h.e_6.h
    R : Type u_1
    inst✝³ : StrictOrderedCommSemiring R
    M : Type u_2
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : IsEmpty ι
    f : LinearEquiv (RingHom.id R) M M
    g : AlternatingMap R M R ι
    hg : Ne g 0
    i : ι → M
    ⊢ Eq (fun i_1 => ↑f.symm (i i_1)) i
  -/
  simp only [LinearEquiv.coe_coe, eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


@[simp]
protected theorem Orientation.map_neg {ι : Type*} (f : M ≃ₗ[R] N) (x : Orientation R M ι) :
    Orientation.map ι f (-x) = -Orientation.map ι f x :=
  Module.Ray.map_neg _ x


@[simp]
protected theorem Orientation.reindex_neg {ι ι' : Type*} (e : ι ≃ ι') (x : Orientation R M ι) :
    Orientation.reindex R M e (-x) = -Orientation.reindex R M e x :=
  Module.Ray.map_neg _ x


/-- The value of `Orientation.map` when the index type has the cardinality of a basis, in terms
of `f.det`. -/
theorem map_orientation_eq_det_inv_smul [Finite ι] (e : Basis ι R M) (x : Orientation R M ι)
    (f : M ≃ₗ[R] M) : Orientation.map ι f x = (LinearEquiv.det f)⁻¹ • x := by
  /-
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : Finite ι
    e : Basis ι R M
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    ⊢ Eq ((Orientation.map ι f) x) (HSMul.hSMul (Inv.inv (LinearEquiv.det f)) x)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : Finite ι
    e : Basis ι R M
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    val✝ : Fintype ι
    ⊢ Eq ((Orientation.map ι f) x) (HSMul.hSMul (Inv.inv (LinearEquiv.det f)) x)
  -/
  letI := Classical.decEq ι
  /-
    case intro
    R : Type u_1
    inst✝³ : StrictOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_4
    inst✝ : Finite ι
    e : Basis ι R M
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    val✝ : Fintype ι
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq ((Orientation.map ι f) x) (HSMul.hSMul (Inv.inv (LinearEquiv.det f)) x)
  -/
  induction' x using Module.Ray.ind with g hg
  rw [Orientation.map_apply, smul_rayOfNeZero, ray_eq_iff, Units.smul_def,
    (g.compLinearMap f.symm).eq_smul_basis_det e, g.eq_smul_basis_det e,
    AlternatingMap.compLinearMap_apply, AlternatingMap.smul_apply,
    show (fun i ↦ (LinearEquiv.symm f).toLinearMap (e i)) = (LinearEquiv.symm f).toLinearMap ∘ e
    by rfl, Basis.det_comp, Basis.det_self, mul_one, smul_eq_mul, mul_comm, mul_smul,
    LinearEquiv.coe_inv_det]


/-- The orientation given by a basis. -/
protected def orientation (e : Basis ι R M) : Orientation R M ι :=
  rayOfNeZero R _ e.det_ne_zero


theorem orientation_map (e : Basis ι R M) (f : M ≃ₗ[R] N) :
    (e.map f).orientation = Orientation.map ι f e.orientation := by
  /-
    R : Type u_1
    inst✝⁶ : StrictOrderedCommRing R
    M : Type u_2
    N : Type u_3
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    f : LinearEquiv (RingHom.id R) M N
    ⊢ Eq (e.map f).orientation ((Orientation.map ι f) e.orientation)
  -/
  simp_rw [Basis.orientation, Orientation.map_apply, Basis.det_map']
  /-
    🎉 no goals
  -/


theorem orientation_reindex (e : Basis ι R M) (eι : ι ≃ ι') :
    (e.reindex eι).orientation = Orientation.reindex R M eι e.orientation := by
  /-
    R : Type u_1
    inst✝⁶ : StrictOrderedCommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    ι : Type u_4
    ι' : Type u_5
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι'
    inst✝ : DecidableEq ι'
    e : Basis ι R M
    eι : Equiv ι ι'
    ⊢ Eq (e.reindex eι).orientation ((Orientation.reindex R M eι) e.orientation)
  -/
  simp_rw [Basis.orientation, Orientation.reindex_apply, Basis.det_reindex']
  /-
    🎉 no goals
  -/


/-- The orientation given by a basis derived using `units_smul`, in terms of the product of those
units. -/
theorem orientation_unitsSMul (e : Basis ι R M) (w : ι → Units R) :
    (e.unitsSMul w).orientation = (∏ i, w i)⁻¹ • e.orientation := by
  rw [Basis.orientation, Basis.orientation, smul_rayOfNeZero, ray_eq_iff,
    e.det.eq_smul_basis_det (e.unitsSMul w), det_unitsSMul_self, Units.smul_def, smul_smul]
  /-
    R : Type u_1
    inst✝⁴ : StrictOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    w : ι → Units R
    ⊢ SameRay R (e.unitsSMul w).det (HSMul.hSMul (HMul.hMul (↑(Inv.inv (Finset.uni …
  -/
  norm_cast
  /-
    R : Type u_1
    inst✝⁴ : StrictOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    w : ι → Units R
    ⊢ SameRay R (e.unitsSMul w).det (HSMul.hSMul (↑(HMul.hMul (Inv.inv (Finset.uni …
  -/
  simp only [inv_mul_cancel, Units.val_one, one_smul]
  /-
    R : Type u_1
    inst✝⁴ : StrictOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    w : ι → Units R
    ⊢ SameRay R (e.unitsSMul w).det (e.unitsSMul w).det
  -/
  exact SameRay.rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem orientation_isEmpty [IsEmpty ι] (b : Basis ι R M) :
    b.orientation = positiveOrientation := by
  /-
    R : Type u_1
    inst✝⁵ : StrictOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : IsEmpty ι
    b : Basis ι R M
    ⊢ Eq b.orientation Module.Oriented.positiveOrientation
  -/
  rw [Basis.orientation]
  /-
    R : Type u_1
    inst✝⁵ : StrictOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : IsEmpty ι
    b : Basis ι R M
    ⊢ Eq (rayOfNeZero R b.det ⋯) Module.Oriented.positiveOrientation
  -/
  congr
  /-
    case e_v
    R : Type u_1
    inst✝⁵ : StrictOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : IsEmpty ι
    b : Basis ι R M
    ⊢ Eq b.det (AlternatingMap.constLinearEquivOfIsEmpty 1)
  -/
  exact b.det_isEmpty
  /-
    🎉 no goals
  -/


/-- A module `M` over a linearly ordered commutative ring has precisely two "orientations" with
respect to an empty index type. (Note that these are only orientations of `M` of in the conventional
mathematical sense if `M` is zero-dimensional.) -/
theorem eq_or_eq_neg_of_isEmpty [IsEmpty ι] (o : Orientation R M ι) :
    o = positiveOrientation ∨ o = -positiveOrientation := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    o : Orientation R M ι
    ⊢ Or (Eq o Module.Oriented.positiveOrientation) (Eq o (Neg.neg Module.Oriented …
  -/
  induction' o using Module.Ray.ind with x hx
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    ⊢ Or (Eq (rayOfNeZero R x hx) Module.Oriented.positiveOrientation) (Eq (rayOfN …
  -/
  dsimp [positiveOrientation]
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    ⊢ Or (Eq (rayOfNeZero R x hx) (rayOfNeZero R (AlternatingMap.constOfIsEmpty R  …
  -/
  simp only [ray_eq_iff, sameRay_neg_swap]
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    ⊢ Or (SameRay R x (AlternatingMap.constOfIsEmpty R M ι 1)) (SameRay R x (Neg.n …
  -/
  rw [sameRay_or_sameRay_neg_iff_not_linearIndependent]
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    ⊢ Not (LinearIndependent R (Matrix.vecCons x (Matrix.vecCons (AlternatingMap.c …
  -/
  intro h
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons (AlternatingMap.cons …
    ⊢ False
  -/
  set f : (M [⋀^ι]→ₗ[R] R) ≃ₗ[R] R := AlternatingMap.constLinearEquivOfIsEmpty.symm
  have H : LinearIndependent R ![f x, 1] := by
    convert h.map' f.toLinearMap f.ker
    ext i
    fin_cases i <;> simp [f]
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons (AlternatingMap.cons …
    f : LinearEquiv (RingHom.id R) (AlternatingMap R M R ι) R := AlternatingMap.co …
    H : LinearIndependent R (Matrix.vecCons (f x) (Matrix.vecCons 1 Matrix.vecEmpt …
    ⊢ False
  -/
  rw [linearIndependent_iff'] at H
  /-
    case h
    R : Type u_1
    inst✝³ : LinearOrderedCommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : IsEmpty ι
    x : AlternatingMap R M R ι
    hx : Ne x 0
    h : LinearIndependent R (Matrix.vecCons x (Matrix.vecCons (AlternatingMap.cons …
    f : LinearEquiv (RingHom.id R) (AlternatingMap R M R ι) R := AlternatingMap.co …
    H : ∀ (s : Finset (Fin (Nat.succ 0).succ)) (g : Fin (Nat.succ 0).succ → R), Eq …
    ⊢ False
  -/
  simpa using H Finset.univ ![1, -f x] (by simp [Fin.sum_univ_succ]) 0 (by simp)
  /-
    🎉 no goals
  -/


/-- The orientations given by two bases are equal if and only if the determinant of one basis
with respect to the other is positive. -/
theorem orientation_eq_iff_det_pos (e₁ e₂ : Basis ι R M) :
    e₁.orientation = e₂.orientation ↔ 0 < e₁.det e₂ :=
  calc
    e₁.orientation = e₂.orientation ↔ SameRay R e₁.det e₂.det := ray_eq_iff _ _
                                                    /-
                                                      R : Type u_1
                                                      inst✝⁴ : LinearOrderedCommRing R
                                                      M : Type u_2
                                                      inst✝³ : AddCommGroup M
                                                      inst✝² : Module R M
                                                      ι : Type u_3
                                                      inst✝¹ : Fintype ι
                                                      inst✝ : DecidableEq ι
                                                      e₁ e₂ : Basis ι R M
                                                      ⊢ Iff (SameRay R e₁.det e₂.det) (SameRay R (HSMul.hSMul (e₁.det ⇑e₂) e₂.det) e …
                                                    -/
    _ ↔ SameRay R (e₁.det e₂ • e₂.det) e₂.det := by rw [← e₁.det.eq_smul_basis_det e₂]
                                                    /-
                                                      🎉 no goals
                                                    -/
    _ ↔ 0 < e₁.det e₂ := sameRay_smul_left_iff_of_ne e₂.det_ne_zero (e₁.isUnit_det e₂).ne_zero


/-- Given a basis, any orientation equals the orientation given by that basis or its negation. -/
theorem orientation_eq_or_eq_neg (e : Basis ι R M) (x : Orientation R M ι) :
    x = e.orientation ∨ x = -e.orientation := by
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    x : Orientation R M ι
    ⊢ Or (Eq x e.orientation) (Eq x (Neg.neg e.orientation))
  -/
  induction' x using Module.Ray.ind with x hx
  /-
    case h
    R : Type u_1
    inst✝⁴ : LinearOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    x : AlternatingMap R M R ι
    hx : Ne x 0
    ⊢ Or (Eq (rayOfNeZero R x hx) e.orientation) (Eq (rayOfNeZero R x hx) (Neg.neg …
  -/
  rw [← x.map_basis_ne_zero_iff e] at hx
  rwa [Basis.orientation, ray_eq_iff, neg_rayOfNeZero, ray_eq_iff, x.eq_smul_basis_det e,
    sameRay_neg_smul_left_iff_of_ne e.det_ne_zero hx, sameRay_smul_left_iff_of_ne e.det_ne_zero hx,
    lt_or_lt_iff_ne, ne_comm]


/-- Given a basis, an orientation equals the negation of that given by that basis if and only
if it does not equal that given by that basis. -/
theorem orientation_ne_iff_eq_neg (e : Basis ι R M) (x : Orientation R M ι) :
    x ≠ e.orientation ↔ x = -e.orientation :=
  ⟨fun h => (e.orientation_eq_or_eq_neg x).resolve_left h, fun h =>
    h.symm ▸ (Module.Ray.ne_neg_self e.orientation).symm⟩


/-- Composing a basis with a linear equiv gives the same orientation if and only if the
determinant is positive. -/
theorem orientation_comp_linearEquiv_eq_iff_det_pos (e : Basis ι R M) (f : M ≃ₗ[R] M) :
    (e.map f).orientation = e.orientation ↔ 0 < LinearMap.det (f : M →ₗ[R] M) := by
  rw [orientation_map, e.map_orientation_eq_det_inv_smul, units_inv_smul, units_smul_eq_self_iff,
    LinearEquiv.coe_det]


/-- Composing a basis with a linear equiv gives the negation of that orientation if and only if
the determinant is negative. -/
theorem orientation_comp_linearEquiv_eq_neg_iff_det_neg (e : Basis ι R M) (f : M ≃ₗ[R] M) :
    (e.map f).orientation = -e.orientation ↔ LinearMap.det (f : M →ₗ[R] M) < 0 := by
  rw [orientation_map, e.map_orientation_eq_det_inv_smul, units_inv_smul, units_smul_eq_neg_iff,
    LinearEquiv.coe_det]


/-- Negating a single basis vector (represented using `units_smul`) negates the corresponding
orientation. -/
@[simp]
theorem orientation_neg_single (e : Basis ι R M) (i : ι) :
    (e.unitsSMul (Function.update 1 i (-1))).orientation = -e.orientation := by
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    i : ι
    ⊢ Eq (e.unitsSMul (Function.update 1 i (-1))).orientation (Neg.neg e.orientati …
  -/
  rw [orientation_unitsSMul, Finset.prod_update_of_mem (Finset.mem_univ _)]
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedCommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    e : Basis ι R M
    i : ι
    ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul (-1) ((SDiff.sdiff Finset.univ (Singleto …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a basis and an orientation, return a basis giving that orientation: either the original
basis, or one constructed by negating a single (arbitrary) basis vector. -/
def adjustToOrientation [Nonempty ι] (e : Basis ι R M) (x : Orientation R M ι) :
    Basis ι R M :=
  haveI := Classical.decEq (Orientation R M ι)
  if e.orientation = x then e else e.unitsSMul (Function.update 1 (Classical.arbitrary ι) (-1))


/-- `adjust_to_orientation` gives a basis with the required orientation. -/
@[simp]
theorem orientation_adjustToOrientation [Nonempty ι] (e : Basis ι R M)
    (x : Orientation R M ι) : (e.adjustToOrientation x).orientation = x := by
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    ⊢ Eq (e.adjustToOrientation x).orientation x
  -/
  rw [adjustToOrientation]
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    ⊢ Eq (ite (Eq e.orientation x) e (e.unitsSMul (Function.update 1 (Classical.ar …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h : Eq e.orientation x
      ⊢ Eq e.orientation x
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h : Not (Eq e.orientation x)
      ⊢ Eq (e.unitsSMul (Function.update 1 (Classical.arbitrary ι) (-1))).orientatio …
    -/
  · rw [orientation_neg_single, eq_comm, ← orientation_ne_iff_eq_neg, ne_comm]
    /-
      case neg
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h : Not (Eq e.orientation x)
      ⊢ Ne e.orientation x
    -/
    exact h
    /-
      🎉 no goals
    -/


/-- Every basis vector from `adjust_to_orientation` is either that from the original basis or its
negation. -/
theorem adjustToOrientation_apply_eq_or_eq_neg [Nonempty ι] (e : Basis ι R M)
    (x : Orientation R M ι) (i : ι) :
    e.adjustToOrientation x i = e i ∨ e.adjustToOrientation x i = -e i := by
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    i : ι
    ⊢ Or (Eq ((e.adjustToOrientation x) i) (e i)) (Eq ((e.adjustToOrientation x) i …
  -/
  rw [adjustToOrientation]
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    i : ι
    ⊢ Or (Eq ((ite (Eq e.orientation x) e (e.unitsSMul (Function.update 1 (Classic …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      i : ι
      h : Eq e.orientation x
      ⊢ Or (Eq (e i) (e i)) (Eq (e i) (Neg.neg (e i)))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      i : ι
      h : Not (Eq e.orientation x)
      ⊢ Or (Eq ((e.unitsSMul (Function.update 1 (Classical.arbitrary ι) (-1))) i) (e …
    -/
                                                /-
                                                  🎉 no goals
                                                -/
  · by_cases hi : i = Classical.arbitrary ι <;> simp [unitsSMul_apply, hi]
                                                /-
                                                  🎉 no goals
                                                -/


theorem det_adjustToOrientation [Nonempty ι] (e : Basis ι R M)
    (x : Orientation R M ι) :
    (e.adjustToOrientation x).det = e.det ∨ (e.adjustToOrientation x).det = -e.det := by
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    ⊢ Or (Eq (e.adjustToOrientation x).det e.det) (Eq (e.adjustToOrientation x).de …
  -/
  dsimp [Basis.adjustToOrientation]
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    ⊢ Or (Eq (ite (Eq e.orientation x) e (e.unitsSMul (Function.update 1 (Classica …
  -/
  split_ifs
    /-
      case pos
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h✝ : Eq e.orientation x
      ⊢ Or (Eq e.det e.det) (Eq e.det (Neg.neg e.det))
    -/
  · left
    /-
      case pos.h
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h✝ : Eq e.orientation x
      ⊢ Eq e.det e.det
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h✝ : Not (Eq e.orientation x)
      ⊢ Or (Eq (e.unitsSMul (Function.update 1 (Classical.arbitrary ι) (-1))).det e. …
    -/
  · right
    simp only [e.det_unitsSMul, ne_eq, Finset.mem_univ, Finset.prod_update_of_mem, not_true,
      Pi.one_apply, Finset.prod_const_one, mul_one, inv_neg', inv_one, Units.val_neg, Units.val_one]
    /-
      case neg.h
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h✝ : Not (Eq e.orientation x)
      ⊢ Eq (HSMul.hSMul (-1) e.det) (Neg.neg e.det)
    -/
    ext
    /-
      case neg.h.H
      R : Type u_1
      inst✝⁵ : LinearOrderedCommRing R
      M : Type u_2
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      ι : Type u_3
      inst✝² : Fintype ι
      inst✝¹ : DecidableEq ι
      inst✝ : Nonempty ι
      e : Basis ι R M
      x : Orientation R M ι
      h✝ : Not (Eq e.orientation x)
      x✝ : ι → M
      ⊢ Eq ((HSMul.hSMul (-1) e.det) x✝) ((Neg.neg e.det) x✝)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem abs_det_adjustToOrientation [Nonempty ι] (e : Basis ι R M)
    (x : Orientation R M ι) (v : ι → M) : |(e.adjustToOrientation x).det v| = |e.det v| := by
  /-
    R : Type u_1
    inst✝⁵ : LinearOrderedCommRing R
    M : Type u_2
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    ι : Type u_3
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    inst✝ : Nonempty ι
    e : Basis ι R M
    x : Orientation R M ι
    v : ι → M
    ⊢ Eq (abs ((e.adjustToOrientation x).det v)) (abs (e.det v))
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  cases' e.det_adjustToOrientation x with h h <;> simp [h]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- If the index type has cardinality equal to the finite dimension, any two orientations are
equal or negations. -/
theorem eq_or_eq_neg [FiniteDimensional R M] (x₁ x₂ : Orientation R M ι)
    (h : Fintype.card ι = finrank R M) : x₁ = x₂ ∨ x₁ = -x₂ := by
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedField R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional R M
    x₁ x₂ : Orientation R M ι
    h : Eq (Fintype.card ι) (Module.finrank R M)
    ⊢ Or (Eq x₁ x₂) (Eq x₁ (Neg.neg x₂))
  -/
  have e := (finBasis R M).reindex (Fintype.equivFinOfCardEq h).symm
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedField R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional R M
    x₁ x₂ : Orientation R M ι
    h : Eq (Fintype.card ι) (Module.finrank R M)
    e : Basis ι R M
    ⊢ Or (Eq x₁ x₂) (Eq x₁ (Neg.neg x₂))
  -/
  letI := Classical.decEq ι
  -- Porting note: this needs to be made explicit for the simp below
  have orientation_neg_neg :
      ∀ f : Basis ι R M, - -Basis.orientation f = Basis.orientation f := by
    #adaptation_note
    /-- `set_option maxSynthPendingDepth 2` required after https://github.com/leanprover/lean4/pull/4119 -/
    set_option maxSynthPendingDepth 2 in simp
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedField R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional R M
    x₁ x₂ : Orientation R M ι
    h : Eq (Fintype.card ι) (Module.finrank R M)
    e : Basis ι R M
    this : DecidableEq ι := Classical.decEq ι
    orientation_neg_neg : ∀ (f : Basis ι R M), Eq (Neg.neg (Neg.neg f.orientation) …
    ⊢ Or (Eq x₁ x₂) (Eq x₁ (Neg.neg x₂))
  -/
  rcases e.orientation_eq_or_eq_neg x₁ with (h₁ | h₁) <;>
    /-
      case inl
      R : Type u_1
      inst✝⁴ : LinearOrderedField R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : FiniteDimensional R M
      x₁ x₂ : Orientation R M ι
      h : Eq (Fintype.card ι) (Module.finrank R M)
      e : Basis ι R M
      this : DecidableEq ι := Classical.decEq ι
      orientation_neg_neg : ∀ (f : Basis ι R M), Eq (Neg.neg (Neg.neg f.orientation) …
      h₁ : Eq x₁ e.orientation
      ⊢ Or (Eq x₁ x₂) (Eq x₁ (Neg.neg x₂))
    -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
    rcases e.orientation_eq_or_eq_neg x₂ with (h₂ | h₂) <;> simp [h₁, h₂, orientation_neg_neg]
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- If the index type has cardinality equal to the finite dimension, an orientation equals the
negation of another orientation if and only if they are not equal. -/
theorem ne_iff_eq_neg [FiniteDimensional R M] (x₁ x₂ : Orientation R M ι)
    (h : Fintype.card ι = finrank R M) : x₁ ≠ x₂ ↔ x₁ = -x₂ :=
  ⟨fun hn => (eq_or_eq_neg x₁ x₂ h).resolve_left hn, fun he =>
    he.symm ▸ (Module.Ray.ne_neg_self x₂).symm⟩


/-- The value of `Orientation.map` when the index type has cardinality equal to the finite
dimension, in terms of `f.det`. -/
theorem map_eq_det_inv_smul [FiniteDimensional R M] (x : Orientation R M ι) (f : M ≃ₗ[R] M)
    (h : Fintype.card ι = finrank R M) : Orientation.map ι f x = (LinearEquiv.det f)⁻¹ • x :=
  haveI e := (finBasis R M).reindex (Fintype.equivFinOfCardEq h).symm
  e.map_orientation_eq_det_inv_smul x f


/-- If the index type has cardinality equal to the finite dimension, composing an alternating
map with the same linear equiv on each argument gives the same orientation if and only if the
determinant is positive. -/
theorem map_eq_iff_det_pos [FiniteDimensional R M] (x : Orientation R M ι) (f : M ≃ₗ[R] M)
    (h : Fintype.card ι = finrank R M) :
    Orientation.map ι f x = x ↔ 0 < LinearMap.det (f : M →ₗ[R] M) := by
  /-
    R : Type u_1
    inst✝⁴ : LinearOrderedField R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional R M
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    h : Eq (Fintype.card ι) (Module.finrank R M)
    ⊢ Iff (Eq ((Orientation.map ι f) x) x) (LT.lt 0 (LinearMap.det ↑f))
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      R : Type u_1
      inst✝⁴ : LinearOrderedField R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : FiniteDimensional R M
      x : Orientation R M ι
      f : LinearEquiv (RingHom.id R) M M
      h : Eq (Fintype.card ι) (Module.finrank R M)
      h✝ : IsEmpty ι
      ⊢ Iff (Eq ((Orientation.map ι f) x) x) (LT.lt 0 (LinearMap.det ↑f))
    -/
  · have H : finrank R M = 0 := h.symm.trans Fintype.card_eq_zero
    /-
      case inl
      R : Type u_1
      inst✝⁴ : LinearOrderedField R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      ι : Type u_3
      inst✝¹ : Fintype ι
      inst✝ : FiniteDimensional R M
      x : Orientation R M ι
      f : LinearEquiv (RingHom.id R) M M
      h : Eq (Fintype.card ι) (Module.finrank R M)
      h✝ : IsEmpty ι
      H : Eq (Module.finrank R M) 0
      ⊢ Iff (Eq ((Orientation.map ι f) x) x) (LT.lt 0 (LinearMap.det ↑f))
    -/
    simp [LinearMap.det_eq_one_of_finrank_eq_zero H]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝⁴ : LinearOrderedField R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_3
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional R M
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    h : Eq (Fintype.card ι) (Module.finrank R M)
    h✝ : Nonempty ι
    ⊢ Iff (Eq ((Orientation.map ι f) x) x) (LT.lt 0 (LinearMap.det ↑f))
  -/
  rw [map_eq_det_inv_smul _ _ h, units_inv_smul, units_smul_eq_self_iff, LinearEquiv.coe_det]
  /-
    🎉 no goals
  -/


/-- If the index type has cardinality equal to the finite dimension, composing an alternating
map with the same linear equiv on each argument gives the negation of that orientation if and
only if the determinant is negative. -/
theorem map_eq_neg_iff_det_neg (x : Orientation R M ι) (f : M ≃ₗ[R] M)
    (h : Fintype.card ι = finrank R M) :
    Orientation.map ι f x = -x ↔ LinearMap.det (f : M →ₗ[R] M) < 0 := by
  /-
    R : Type u_1
    inst✝³ : LinearOrderedField R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : Fintype ι
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    h : Eq (Fintype.card ι) (Module.finrank R M)
    ⊢ Iff (Eq ((Orientation.map ι f) x) (Neg.neg x)) (LT.lt (LinearMap.det ↑f) 0)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      R : Type u_1
      inst✝³ : LinearOrderedField R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type u_3
      inst✝ : Fintype ι
      x : Orientation R M ι
      f : LinearEquiv (RingHom.id R) M M
      h : Eq (Fintype.card ι) (Module.finrank R M)
      h✝ : IsEmpty ι
      ⊢ Iff (Eq ((Orientation.map ι f) x) (Neg.neg x)) (LT.lt (LinearMap.det ↑f) 0)
    -/
  · have H : finrank R M = 0 := h.symm.trans Fintype.card_eq_zero
    /-
      case inl
      R : Type u_1
      inst✝³ : LinearOrderedField R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      ι : Type u_3
      inst✝ : Fintype ι
      x : Orientation R M ι
      f : LinearEquiv (RingHom.id R) M M
      h : Eq (Fintype.card ι) (Module.finrank R M)
      h✝ : IsEmpty ι
      H : Eq (Module.finrank R M) 0
      ⊢ Iff (Eq ((Orientation.map ι f) x) (Neg.neg x)) (LT.lt (LinearMap.det ↑f) 0)
    -/
    simp [LinearMap.det_eq_one_of_finrank_eq_zero H, Module.Ray.ne_neg_self x]
    /-
      🎉 no goals
    -/
  have H : 0 < finrank R M := by
    rw [← h]
    exact Fintype.card_pos
  /-
    case inr
    R : Type u_1
    inst✝³ : LinearOrderedField R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : Fintype ι
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    h : Eq (Fintype.card ι) (Module.finrank R M)
    h✝ : Nonempty ι
    H : LT.lt 0 (Module.finrank R M)
    ⊢ Iff (Eq ((Orientation.map ι f) x) (Neg.neg x)) (LT.lt (LinearMap.det ↑f) 0)
  -/
  haveI : FiniteDimensional R M := of_finrank_pos H
  /-
    case inr
    R : Type u_1
    inst✝³ : LinearOrderedField R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u_3
    inst✝ : Fintype ι
    x : Orientation R M ι
    f : LinearEquiv (RingHom.id R) M M
    h : Eq (Fintype.card ι) (Module.finrank R M)
    h✝ : Nonempty ι
    H : LT.lt 0 (Module.finrank R M)
    this : FiniteDimensional R M
    ⊢ Iff (Eq ((Orientation.map ι f) x) (Neg.neg x)) (LT.lt (LinearMap.det ↑f) 0)
  -/
  rw [map_eq_det_inv_smul _ _ h, units_inv_smul, units_smul_eq_neg_iff, LinearEquiv.coe_det]
  /-
    🎉 no goals
  -/


/-- If the index type has cardinality equal to the finite dimension, a basis with the given
orientation. -/
def someBasis [Nonempty ι] [DecidableEq ι] [FiniteDimensional R M] (x : Orientation R M ι)
    (h : Fintype.card ι = finrank R M) : Basis ι R M :=
  ((finBasis R M).reindex (Fintype.equivFinOfCardEq h).symm).adjustToOrientation x


/-- `some_basis` gives a basis with the required orientation. -/
@[simp]
theorem someBasis_orientation [Nonempty ι] [DecidableEq ι] [FiniteDimensional R M]
    (x : Orientation R M ι) (h : Fintype.card ι = finrank R M) : (x.someBasis h).orientation = x :=
  Basis.orientation_adjustToOrientation _ _


