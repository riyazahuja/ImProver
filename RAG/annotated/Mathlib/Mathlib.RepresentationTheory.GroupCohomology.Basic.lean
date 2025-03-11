/-- The complex `Hom(P, A)`, where `P` is the standard resolution of `k` as a trivial `k`-linear
`G`-representation. -/
abbrev linearYonedaObjResolution (A : Rep k G) : CochainComplex (ModuleCat.{u} k) ℕ :=
  (groupCohomology.resolution k G).linearYonedaObj k A


theorem linearYonedaObjResolution_d_apply {A : Rep k G} (i j : ℕ) (x : (resolution k G).X i ⟶ A) :
    (linearYonedaObjResolution A).d i j x = (resolution k G).d j i ≫ x :=
  rfl


/-- The differential in the complex of inhomogeneous cochains used to
calculate group cohomology. -/
@[simps]
def d [Monoid G] (n : ℕ) (A : Rep k G) : ((Fin n → G) → A) →ₗ[k] (Fin (n + 1) → G) → A where
  toFun f g :=
    A.ρ (g 0) (f fun i => g i.succ) +
      Finset.univ.sum fun j : Fin (n + 1) =>
        (-1 : k) ^ ((j : ℕ) + 1) • f (Fin.contractNth j (· * ·) g)
  map_add' f g := by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Monoid G
      n : Nat
      A : Rep k G
      f g : (Fin n → G) → CoeSort.coe A
      ⊢ Eq ((fun f g => HAdd.hAdd ((A.ρ (g 0)) (f fun i => g i.succ)) (Finset.univ.s …
    -/
    ext x
/- Porting note: changed from `simp only` which needed extra heartbeats -/
    /-
      case h
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Monoid G
      n : Nat
      A : Rep k G
      f g : (Fin n → G) → CoeSort.coe A
      x : Fin (HAdd.hAdd n 1) → G
      ⊢ Eq ((fun f g => HAdd.hAdd ((A.ρ (g 0)) (f fun i => g i.succ)) (Finset.univ.s …
    -/
    simp_rw [Pi.add_apply, map_add, smul_add, Finset.sum_add_distrib, add_add_add_comm]
    /-
      🎉 no goals
    -/
  map_smul' r f := by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Monoid G
      n : Nat
      A : Rep k G
      r : k
      f : (Fin n → G) → CoeSort.coe A
      ⊢ Eq ({ toFun := fun f g => HAdd.hAdd ((A.ρ (g 0)) (f fun i => g i.succ)) (Fin …
    -/
    ext x
/- Porting note: changed from `simp only` which needed extra heartbeats -/
    simp_rw [Pi.smul_apply, RingHom.id_apply, map_smul, smul_add, Finset.smul_sum, ← smul_assoc,
      smul_eq_mul, mul_comm r]


/-- The theorem that our isomorphism `Fun(Gⁿ, A) ≅ Hom(k[Gⁿ⁺¹], A)` (where the righthand side is
morphisms in `Rep k G`) commutes with the differentials in the complex of inhomogeneous cochains
and the homogeneous `linearYonedaObjResolution`. -/
@[nolint checkType] theorem d_eq :
    d n A =
      ((diagonalHomEquiv n A).toModuleIso.inv ≫
        (linearYonedaObjResolution A).d n (n + 1) ≫
          (diagonalHomEquiv (n + 1) A).toModuleIso.hom).hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    ⊢ Eq (inhomogeneousCochains.d n A) (CategoryTheory.CategoryStruct.comp (Rep.di …
  -/
  ext f g
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
  simp only [ModuleCat.coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    LinearEquiv.toModuleIso_inv, linearYonedaObjResolution_d_apply, LinearEquiv.toModuleIso_hom,
    diagonalHomEquiv_apply, Action.comp_hom, Resolution.d_eq k G n,
    Resolution.d_of (Fin.partialProd g), LinearMap.map_sum,
    ← Finsupp.smul_single_one _ ((-1 : k) ^ _), map_smul, d_apply]
  simp only [@Fin.sum_univ_succ _ _ (n + 1), Fin.val_zero, pow_zero, one_smul, Fin.succAbove_zero,
    diagonalHomEquiv_symm_apply f (Fin.partialProd g ∘ @Fin.succ (n + 1)), Function.comp_apply,
    Fin.partialProd_succ, Fin.castSucc_zero, Fin.partialProd_zero, one_mul]
  congr 1
  · congr
    ext
    have := Fin.partialProd_right_inv g (Fin.castSucc x)
    simp only [mul_inv_rev, Fin.castSucc_fin_succ] at *
    rw [mul_assoc, ← mul_assoc _ _ (g x.succ), this, inv_mul_cancel_left]
  · exact Finset.sum_congr rfl fun j hj => by
      rw [diagonalHomEquiv_symm_partialProd_succ, Fin.val_succ] -/
  -- https://github.com/leanprover-community/mathlib4/issues/5026
  -- https://github.com/leanprover-community/mathlib4/issues/5164
  change d n A f g = diagonalHomEquiv (n + 1) A
    ((resolution k G).d (n + 1) n ≫ (diagonalHomEquiv n A).symm f) g
  rw [diagonalHomEquiv_apply, Action.comp_hom, ModuleCat.hom_comp, LinearMap.comp_apply,
    resolution.d_eq]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    g : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq ((inhomogeneousCochains.d n A) f g) (((Rep.diagonalHomEquiv n A).symm f). …
  -/
  erw [resolution.d_of (Fin.partialProd g)]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    g : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq ((inhomogeneousCochains.d n A) f g) (((Rep.diagonalHomEquiv n A).symm f). …
  -/
  simp only [map_sum, ← Finsupp.smul_single_one _ ((-1 : k) ^ _)]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [d_apply, @Fin.sum_univ_succ _ _ (n + 1), Fin.val_zero, pow_zero, one_smul,
    Fin.succAbove_zero, diagonalHomEquiv_symm_apply f (Fin.partialProd g ∘ @Fin.succ (n + 1))]
  simp_rw [Function.comp_apply, Fin.partialProd_succ, Fin.castSucc_zero,
    Fin.partialProd_zero, one_mul]
  /-
    case h.h
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    g : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq (HAdd.hAdd ((A.ρ (g 0)) (f fun i => g i.succ)) (Finset.univ.sum fun j =>  …
  -/
  rcongr x
    /-
      case h.h.e_a.h.e_6.h.e_a.h
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      A : Rep k G
      f : (Fin n → G) → CoeSort.coe A
      g : Fin (HAdd.hAdd n 1) → G
      x : Fin n
      ⊢ Eq (g x.succ) (HMul.hMul (Inv.inv (HMul.hMul (Fin.partialProd g x.castSucc.c …
    -/
  · have := Fin.partialProd_right_inv g (Fin.castSucc x)
    /-
      case h.h.e_a.h.e_6.h.e_a.h
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      A : Rep k G
      f : (Fin n → G) → CoeSort.coe A
      g : Fin (HAdd.hAdd n 1) → G
      x : Fin n
      this : Eq (HMul.hMul (Inv.inv (Fin.partialProd g x.castSucc.castSucc)) (Fin.pa …
      ⊢ Eq (g x.succ) (HMul.hMul (Inv.inv (HMul.hMul (Fin.partialProd g x.castSucc.c …
    -/
    simp only [mul_inv_rev, Fin.castSucc_fin_succ] at this ⊢
    /-
      case h.h.e_a.h.e_6.h.e_a.h
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      A : Rep k G
      f : (Fin n → G) → CoeSort.coe A
      g : Fin (HAdd.hAdd n 1) → G
      x : Fin n
      this : Eq (HMul.hMul (Inv.inv (Fin.partialProd g x.castSucc.castSucc)) (Fin.pa …
      ⊢ Eq (g x.succ) (HMul.hMul (HMul.hMul (Inv.inv (g x.castSucc)) (Inv.inv (Fin.p …
    -/
    rw [mul_assoc, ← mul_assoc _ _ (g x.succ), this, inv_mul_cancel_left]
    /-
      🎉 no goals
    -/
  · -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case h.h.e_a.e_f.h
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      A : Rep k G
      f : (Fin n → G) → CoeSort.coe A
      g : Fin (HAdd.hAdd n 1) → G
      x : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HAdd.hAdd (↑x) 1)) (f (x.contractNth (fun x …
    -/
    erw [map_smul, diagonalHomEquiv_symm_partialProd_succ, Fin.val_succ]
    /-
      🎉 no goals
    -/


/-- Given a `k`-linear `G`-representation `A`, this is the complex of inhomogeneous cochains
$$0 \to \mathrm{Fun}(G^0, A) \to \mathrm{Fun}(G^1, A) \to \mathrm{Fun}(G^2, A) \to \dots$$
which calculates the group cohomology of `A`. -/
noncomputable abbrev inhomogeneousCochains : CochainComplex (ModuleCat k) ℕ :=
  CochainComplex.of (fun n => ModuleCat.of k ((Fin n → G) → A))
    (fun n => ModuleCat.ofHom (inhomogeneousCochains.d n A)) fun n => by
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
    ext x y
    have := LinearMap.ext_iff.1 ((linearYonedaObjResolution A).d_comp_d n (n + 1) (n + 2))
    simp only [ModuleCat.coe_comp, Function.comp_apply] at this
    simp only [ModuleCat.coe_comp, Function.comp_apply, d_eq, LinearEquiv.toModuleIso_hom,
      LinearEquiv.toModuleIso_inv, LinearEquiv.coe_coe, LinearEquiv.symm_apply_apply, this,
      LinearMap.zero_apply, map_zero, Pi.zero_apply] -/
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Group G
      A : Rep k G
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun n => ModuleCat.ofHom (inhomogen …
    -/
    ext x
    have : ∀ x, _ = (0 : _ →ₗ[_] _) x := LinearMap.ext_iff.1 (ModuleCat.hom_ext_iff.mp
      ((linearYonedaObjResolution A).d_comp_d n (n + 1) (n + 2)))
    /-
      case hf.h.h
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Group G
      A : Rep k G
      n : Nat
      x : ↑((fun n => ModuleCat.of k ((Fin n → G) → CoeSort.coe A)) n)
      x✝ : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      this : ∀ (x : ↑((groupCohomology.linearYonedaObjResolution A).X n)), Eq ((Cate …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun n => ModuleCat.ofHom (inhomoge …
    -/
    simp only [ModuleCat.hom_comp, LinearMap.comp_apply] at this
    /-
      case hf.h.h
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Group G
      A : Rep k G
      n : Nat
      x : ↑((fun n => ModuleCat.of k ((Fin n → G) → CoeSort.coe A)) n)
      x✝ : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      this : ∀ (x : ↑((groupCohomology.linearYonedaObjResolution A).X n)), Eq (((gro …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun n => ModuleCat.ofHom (inhomoge …
    -/
    dsimp only
    simp only [d_eq, LinearEquiv.toModuleIso_inv_hom, LinearEquiv.toModuleIso_hom_hom,
      ModuleCat.hom_comp, LinearMap.comp_apply, LinearEquiv.coe_coe, ModuleCat.hom_zero]
    /- Porting note: I can see I need to rewrite `LinearEquiv.coe_coe` twice to at
      least reduce the need for `symm_apply_apply` to be an `erw`. However, even `erw` refuses to
      rewrite the second `coe_coe`... -/
    /-
      case hf.h.h
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Group G
      A : Rep k G
      n : Nat
      x : ↑((fun n => ModuleCat.of k ((Fin n → G) → CoeSort.coe A)) n)
      x✝ : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      this : ∀ (x : ↑((groupCohomology.linearYonedaObjResolution A).X n)), Eq (((gro …
      ⊢ Eq ((Rep.diagonalHomEquiv (HAdd.hAdd (HAdd.hAdd n 1) 1) A) (((groupCohomolog …
    -/
    erw [LinearEquiv.symm_apply_apply, this]
    simp only [LinearMap.zero_apply, ChainComplex.linearYonedaObj_X, linearYoneda_obj_obj_carrier,
      map_zero, Pi.zero_apply, LinearMap.zero_apply]
    /-
      case hf.h.h
      k G : Type u
      inst✝¹ : CommRing k
      n✝ : Nat
      inst✝ : Group G
      A : Rep k G
      n : Nat
      x : ↑((fun n => ModuleCat.of k ((Fin n → G) → CoeSort.coe A)) n)
      x✝ : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      this : ∀ (x : ↑((groupCohomology.linearYonedaObjResolution A).X n)), Eq (((gro …
      ⊢ Eq ((Rep.diagonalHomEquiv (HAdd.hAdd (HAdd.hAdd n 1) 1) A) (0 ((Rep.diagonal …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem inhomogeneousCochains.d_def (n : ℕ) :
    (inhomogeneousCochains A).d n (n + 1) = ModuleCat.ofHom (inhomogeneousCochains.d n A) :=
  CochainComplex.of_d _ _ _ _


/-- Given a `k`-linear `G`-representation `A`, the complex of inhomogeneous cochains is isomorphic
to `Hom(P, A)`, where `P` is the standard resolution of `k` as a trivial `G`-representation. -/
def inhomogeneousCochainsIso : inhomogeneousCochains A ≅ linearYonedaObjResolution A := by
  refine HomologicalComplex.Hom.isoOfComponents (fun i =>
    (Rep.diagonalHomEquiv i A).toModuleIso.symm) ?_
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    ⊢ ∀ (i j : Nat), (ComplexShape.up Nat).Rel i j → Eq (CategoryTheory.CategorySt …
  -/
  rintro i j (h : i + 1 = j)
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    i j : Nat
    h : Eq (HAdd.hAdd i 1) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (Rep.diagonalHomEquiv i A) …
  -/
  subst h
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (Rep.diagonalHomEquiv i A) …
  -/
  ext
  simp only [ChainComplex.linearYonedaObj_X, linearYoneda_obj_obj_carrier, CochainComplex.of_x,
    linearYoneda_obj_obj_isAddCommGroup, linearYoneda_obj_obj_isModule, Iso.symm_hom,
    ChainComplex.linearYonedaObj_d, ModuleCat.hom_comp, linearYoneda_obj_map_hom,
    Quiver.Hom.unop_op, LinearEquiv.toModuleIso_inv_hom, LinearMap.coe_comp, Function.comp_apply,
    Linear.leftComp_apply, inhomogeneousCochains.d_def, d_eq, LinearEquiv.toModuleIso_hom_hom,
    ModuleCat.ofHom_comp, Category.assoc, LinearEquiv.comp_coe, LinearEquiv.self_trans_symm,
    LinearEquiv.refl_toLinearMap, LinearMap.id_comp, LinearEquiv.coe_coe]
  /-
    case hf.h
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    i : Nat
    x✝ : ↑((groupCohomology.inhomogeneousCochains A).X i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((groupCohomology.resolution k G).d ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The `n`-cocycles `Zⁿ(G, A)` of a `k`-linear `G`-representation `A`, i.e. the kernel of the
`n`th differential in the complex of inhomogeneous cochains. -/
abbrev cocycles (n : ℕ) : ModuleCat k := (inhomogeneousCochains A).cycles n


/-- The natural inclusion of the `n`-cocycles `Zⁿ(G, A)` into the `n`-cochains `Cⁿ(G, A).` -/
abbrev iCocycles (n : ℕ) : cocycles A n ⟶ ModuleCat.of k ((Fin n → G) → A) :=
  (inhomogeneousCochains A).iCycles n


/-- This is the map from `i`-cochains to `j`-cocycles induced by the differential in the complex of
inhomogeneous cochains. -/
abbrev toCocycles (i j : ℕ) : ModuleCat.of k ((Fin i → G) → A) ⟶ cocycles A j :=
  (inhomogeneousCochains A).toCycles i j


/-- The group cohomology of a `k`-linear `G`-representation `A`, as the cohomology of its complex
of inhomogeneous cochains. -/
def groupCohomology [Group G] (A : Rep k G) (n : ℕ) : ModuleCat k :=
  (inhomogeneousCochains A).homology n


/-- The natural map from `n`-cocycles to `n`th group cohomology for a `k`-linear
`G`-representation `A`. -/
abbrev groupCohomologyπ [Group G] (A : Rep k G) (n : ℕ) :
    groupCohomology.cocycles A n ⟶ groupCohomology A n :=
  (inhomogeneousCochains A).homologyπ n


/-- The `n`th group cohomology of a `k`-linear `G`-representation `A` is isomorphic to
`Extⁿ(k, A)` (taken in `Rep k G`), where `k` is a trivial `k`-linear `G`-representation. -/
def groupCohomologyIsoExt [Group G] (A : Rep k G) (n : ℕ) :
    groupCohomology A n ≅ ((Ext k (Rep k G) n).obj (Opposite.op <| Rep.trivial k G k)).obj A :=
  isoOfQuasiIsoAt (HomotopyEquiv.ofIso (inhomogeneousCochainsIso A)).hom n ≪≫
    (extIso k G A n).symm

