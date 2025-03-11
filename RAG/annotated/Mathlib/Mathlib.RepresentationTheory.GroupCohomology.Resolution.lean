local notation "Gⁿ" => Fin n → G


local notation "Gⁿ⁺¹" => Fin (n + 1) → G


/-- An isomorphism of `G`-sets `Gⁿ⁺¹ ≅ G × Gⁿ`, where `G` acts by left multiplication on `Gⁿ⁺¹` and
`G` but trivially on `Gⁿ`. The map sends `(g₀, ..., gₙ) ↦ (g₀, (g₀⁻¹g₁, g₁⁻¹g₂, ..., gₙ₋₁⁻¹gₙ))`,
and the inverse is `(g₀, (g₁, ..., gₙ)) ↦ (g₀, g₀g₁, g₀g₁g₂, ..., g₀g₁...gₙ).` -/
def actionDiagonalSucc (G : Type u) [Group G] :
    ∀ n : ℕ, diagonal G (n + 1) ≅ leftRegular G ⊗ Action.mk (Fin n → G) 1
  | 0 =>
    diagonalOneIsoLeftRegular G ≪≫
      (ρ_ _).symm ≪≫ tensorIso (Iso.refl _) (tensorUnitIso (Equiv.ofUnique PUnit _).toIso)
  | n + 1 =>
    diagonalSucc _ _ ≪≫
      tensorIso (Iso.refl _) (actionDiagonalSucc G n) ≪≫
        leftRegularTensorIso _ _ ≪≫
          tensorIso (Iso.refl _)
            (mkIso (Fin.insertNthEquiv (fun _ => G) 0).toIso fun _ => rfl)


theorem actionDiagonalSucc_hom_apply {G : Type u} [Group G] {n : ℕ} (f : Fin (n + 1) → G) :
    (actionDiagonalSucc G n).hom.hom f = (f 0, fun i => (f (Fin.castSucc i))⁻¹ * f i.succ) := by
  /-
    G : Type u
    inst✝ : Group G
    n : Nat
    f : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G n).hom.hom f) { fst :=  …
  -/
  induction' n with n hn
    /-
      case zero
      G : Type u
      inst✝ : Group G
      f : Fin (HAdd.hAdd 0 1) → G
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G 0).hom.hom f) { fst :=  …
    -/
  · exact Prod.ext rfl (funext fun x => Fin.elim0 x)
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin (HAdd.hAdd n 1) → G), Eq ((groupCohomology.resolution.actionDi …
      f : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G (HAdd.hAdd n 1)).hom.ho …
    -/
  · refine Prod.ext rfl (funext fun x => ?_)
    /- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
    · dsimp only [actionDiagonalSucc]
      simp only [Iso.trans_hom, comp_hom, types_comp_apply, diagonalSucc_hom_hom,
        leftRegularTensorIso_hom_hom, tensorIso_hom, mkIso_hom_hom, Equiv.toIso_hom,
        Action.tensorHom, Equiv.piFinSuccAbove_symm_apply, tensor_apply, types_id_apply,
        tensor_rho, MonoidHom.one_apply, End.one_def, hn fun j : Fin (n + 1) => f j.succ,
        Fin.insertNth_zero']
      refine' Fin.cases (Fin.cons_zero _ _) (fun i => _) x
      · simp only [Fin.cons_succ, mul_left_inj, inv_inj, Fin.castSucc_fin_succ] -/
    /-
      case succ
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin (HAdd.hAdd n 1) → G), Eq ((groupCohomology.resolution.actionDi …
      f : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      x : Fin (HAdd.hAdd n 1)
      ⊢ Eq (((groupCohomology.resolution.actionDiagonalSucc G (HAdd.hAdd n 1)).hom.h …
    -/
    dsimp [actionDiagonalSucc]
    /-
      case succ
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin (HAdd.hAdd n 1) → G), Eq ((groupCohomology.resolution.actionDi …
      f : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      x : Fin (HAdd.hAdd n 1)
      ⊢ Eq ((Fin.insertNthEquiv (fun x => G) 0) ((CategoryTheory.MonoidalCategoryStr …
    -/
    erw [hn (fun (j : Fin (n + 1)) => f j.succ)]
    /-
      case succ
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin (HAdd.hAdd n 1) → G), Eq ((groupCohomology.resolution.actionDi …
      f : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → G
      x : Fin (HAdd.hAdd n 1)
      ⊢ Eq ((Fin.insertNthEquiv (fun x => G) 0) ((CategoryTheory.MonoidalCategoryStr …
    -/
    exact Fin.cases rfl (fun i => rfl) x
    /-
      🎉 no goals
    -/


theorem actionDiagonalSucc_inv_apply {G : Type u} [Group G] {n : ℕ} (g : G) (f : Fin n → G) :
    (actionDiagonalSucc G n).inv.hom (g, f) = (g • Fin.partialProd f : Fin (n + 1) → G) := by
  /-
    G : Type u
    inst✝ : Group G
    n : Nat
    g : G
    f : Fin n → G
    ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G n).inv.hom { fst := g,  …
  -/
  revert g
  /-
    G : Type u
    inst✝ : Group G
    n : Nat
    f : Fin n → G
    ⊢ ∀ (g : G), Eq ((groupCohomology.resolution.actionDiagonalSucc G n).inv.hom { …
  -/
  induction' n with n hn
    /-
      case zero
      G : Type u
      inst✝ : Group G
      f : Fin 0 → G
      ⊢ ∀ (g : G), Eq ((groupCohomology.resolution.actionDiagonalSucc G 0).inv.hom { …
    -/
  · intro g
    /-
      case zero
      G : Type u
      inst✝ : Group G
      f : Fin 0 → G
      g : G
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G 0).inv.hom { fst := g,  …
    -/
    funext (x : Fin 1)
    /-
      case zero.h
      G : Type u
      inst✝ : Group G
      f : Fin 0 → G
      g : G
      x : Fin 1
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G 0).inv.hom { fst := g,  …
    -/
    simp only [Subsingleton.elim x 0, Pi.smul_apply, Fin.partialProd_zero, smul_eq_mul, mul_one]
    /-
      case zero.h
      G : Type u
      inst✝ : Group G
      f : Fin 0 → G
      g : G
      x : Fin 1
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G 0).inv.hom { fst := g,  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
      f : Fin (HAdd.hAdd n 1) → G
      ⊢ ∀ (g : G), Eq ((groupCohomology.resolution.actionDiagonalSucc G (HAdd.hAdd n …
    -/
  · intro g
    /- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
    ext
    dsimp only [actionDiagonalSucc]
    simp only [Iso.trans_inv, comp_hom, hn, diagonalSucc_inv_hom, types_comp_apply, tensorIso_inv,
      Iso.refl_inv, Action.tensorHom, id_hom, tensor_apply, types_id_apply,
      leftRegularTensorIso_inv_hom, tensor_ρ, leftRegular_ρ_apply, Pi.smul_apply, smul_eq_mul]
    refine' Fin.cases _ _ x
    · simp only [Fin.cons_zero, Fin.partialProd_zero, mul_one]
    · intro i
      simpa only [Fin.cons_succ, Pi.smul_apply, smul_eq_mul, Fin.partialProd_succ', mul_assoc] -/
    /-
      case succ
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
      f : Fin (HAdd.hAdd n 1) → G
      g : G
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G (HAdd.hAdd n 1)).inv.ho …
    -/
    funext x
    /-
      case succ.h
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
      f : Fin (HAdd.hAdd n 1) → G
      g : G
      x : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq ((groupCohomology.resolution.actionDiagonalSucc G (HAdd.hAdd n 1)).inv.ho …
    -/
    dsimp [actionDiagonalSucc]
    /-
      case succ.h
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
      f : Fin (HAdd.hAdd n 1) → G
      g : G
      x : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq ((Fin.consEquiv fun a => G) { fst := g, snd := (groupCohomology.resolutio …
    -/
    erw [hn, Fin.consEquiv_apply]
    /-
      case succ.h
      G : Type u
      inst✝ : Group G
      n : Nat
      hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
      f : Fin (HAdd.hAdd n 1) → G
      g : G
      x : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (Fin.cons { fst := g, snd := HSMul.hSMul ((CategoryTheory.Limits.BinaryFa …
    -/
    refine Fin.cases ?_ (fun i => ?_) x
      /-
        case succ.h.refine_1
        G : Type u
        inst✝ : Group G
        n : Nat
        hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
        f : Fin (HAdd.hAdd n 1) → G
        g : G
        x : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ Eq (Fin.cons { fst := g, snd := HSMul.hSMul ((CategoryTheory.Limits.BinaryFa …
      -/
    · simp only [Fin.insertNth_zero, Fin.cons_zero, Fin.partialProd_zero, mul_one]
      /-
        🎉 no goals
      -/
      /-
        case succ.h.refine_2
        G : Type u
        inst✝ : Group G
        n : Nat
        hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
        f : Fin (HAdd.hAdd n 1) → G
        g : G
        x : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (Fin.cons { fst := g, snd := HSMul.hSMul ((CategoryTheory.Limits.BinaryFa …
      -/
    · simp only [Fin.cons_succ, Pi.smul_apply, smul_eq_mul, Fin.partialProd_succ', ← mul_assoc]
      /-
        case succ.h.refine_2
        G : Type u
        inst✝ : Group G
        n : Nat
        hn : ∀ (f : Fin n → G) (g : G), Eq ((groupCohomology.resolution.actionDiagonal …
        f : Fin (HAdd.hAdd n 1) → G
        g : G
        x : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (HMul.hMul ((CategoryTheory.Limits.BinaryFan.mk (CategoryTheory.CategoryS …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- An isomorphism of `k`-linear representations of `G` from `k[Gⁿ⁺¹]` to `k[G] ⊗ₖ k[Gⁿ]` (on
which `G` acts by `ρ(g₁)(g₂ ⊗ x) = (g₁ * g₂) ⊗ x`) sending `(g₀, ..., gₙ)` to
`g₀ ⊗ (g₀⁻¹g₁, g₁⁻¹g₂, ..., gₙ₋₁⁻¹gₙ)`. The inverse sends `g₀ ⊗ (g₁, ..., gₙ)` to
`(g₀, g₀g₁, ..., g₀g₁...gₙ)`. -/
def diagonalSucc (n : ℕ) :
    diagonal k G (n + 1) ≅ leftRegular k G ⊗ trivial k G ((Fin n → G) →₀ k) :=
  (linearization k G).mapIso (actionDiagonalSucc G n) ≪≫
    (Functor.Monoidal.μIso (linearization k G) _ _).symm ≪≫
      tensorIso (Iso.refl _) (linearizationTrivialIso k G (Fin n → G))


theorem diagonalSucc_hom_single (f : Gⁿ⁺¹) (a : k) :
    (diagonalSucc k G n).hom.hom (single f a) =
      single (f 0) 1 ⊗ₜ single (fun i => (f (Fin.castSucc i))⁻¹ * f i.succ) a := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    f : Fin (HAdd.hAdd n 1) → G
    a : k
    ⊢ Eq ((groupCohomology.resolution.diagonalSucc k G n).hom.hom.hom (Finsupp.sin …
  -/
  dsimp [diagonalSucc]
  erw [lmapDomain_apply, mapDomain_single, LinearEquiv.coe_toLinearMap, finsuppTensorFinsupp',
    LinearEquiv.trans_symm, LinearEquiv.trans_apply, lcongr_symm, Equiv.refl_symm]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    f : Fin (HAdd.hAdd n 1) → G
    a : k
    ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Categor …
  -/
  erw [lcongr_single]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    f : Fin (HAdd.hAdd n 1) → G
    a : k
    ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Categor …
  -/
  rw [TensorProduct.lid_symm_apply, actionDiagonalSucc_hom_apply, finsuppTensorFinsupp_symm_single]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    f : Fin (HAdd.hAdd n 1) → G
    a : k
    ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.tensorHom (CategoryTheory.Categor …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem diagonalSucc_inv_single_single (g : G) (f : Gⁿ) (a b : k) :
    (diagonalSucc k G n).inv.hom (Finsupp.single g a ⊗ₜ Finsupp.single f b) =
      single (g • partialProd f) (a * b) := by
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
  dsimp only [diagonalSucc]
  simp only [Iso.trans_inv, Iso.symm_inv, Iso.refl_inv, tensorIso_inv, Action.tensorHom,
    Action.comp_hom, ModuleCat.comp_def, LinearMap.comp_apply, asIso_hom, Functor.mapIso_inv,
    ModuleCat.MonoidalCategory.hom_apply, linearizationTrivialIso_inv_hom_apply,
    linearization_μ_hom, Action.id_hom ((linearization k G).obj _), actionDiagonalSucc_inv_apply,
    ModuleCat.id_apply, LinearEquiv.coe_toLinearMap,
    finsuppTensorFinsupp'_single_tmul_single k (Action.leftRegular G).V,
    linearization_map_hom_single (actionDiagonalSucc G n).inv (g, f) (a * b)] -/
  change mapDomain (actionDiagonalSucc G n).inv.hom
    (lcongr (Equiv.refl (G × (Fin n → G))) (TensorProduct.lid k k)
      (finsuppTensorFinsupp k k k k G (Fin n → G) (single g a ⊗ₜ[k] single f b)))
    = single (g • partialProd f) (a * b)
  rw [finsuppTensorFinsupp_single, lcongr_single, mapDomain_single, Equiv.refl_apply,
    actionDiagonalSucc_inv_apply]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    g : G
    f : Fin n → G
    a b : k
    ⊢ Eq (Finsupp.single (HSMul.hSMul g (Fin.partialProd f)) ((TensorProduct.lid k …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem diagonalSucc_inv_single_left (g : G) (f : Gⁿ →₀ k) (r : k) :
    (diagonalSucc k G n).inv.hom (Finsupp.single g r ⊗ₜ f) =
      Finsupp.lift (Gⁿ⁺¹ →₀ k) k Gⁿ (fun f => single (g • partialProd f) r) f := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    g : G
    f : Finsupp (Fin n → G) k
    r : k
    ⊢ Eq ((groupCohomology.resolution.diagonalSucc k G n).inv.hom.hom (TensorProdu …
  -/
  refine f.induction ?_ ?_
    /-
      case refine_1
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      g : G
      f : Finsupp (Fin n → G) k
      r : k
      ⊢ Eq ((groupCohomology.resolution.diagonalSucc k G n).inv.hom.hom (TensorProdu …
    -/
  · simp only [TensorProduct.tmul_zero, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      g : G
      f : Finsupp (Fin n → G) k
      r : k
      ⊢ ∀ (a : Fin n → G) (b : k) (f : Finsupp (Fin n → G) k), Not (Membership.mem f …
    -/
  · intro a b x _ _ hx
    simp only [lift_apply, smul_single', mul_one, TensorProduct.tmul_add, map_add,
      diagonalSucc_inv_single_single, hx, Finsupp.sum_single_index, mul_comm b,
      zero_mul, single_zero]


theorem diagonalSucc_inv_single_right (g : G →₀ k) (f : Gⁿ) (r : k) :
    (diagonalSucc k G n).inv.hom (g ⊗ₜ Finsupp.single f r) =
      Finsupp.lift _ k G (fun a => single (a • partialProd f) r) g := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    g : Finsupp G k
    f : Fin n → G
    r : k
    ⊢ Eq ((groupCohomology.resolution.diagonalSucc k G n).inv.hom.hom (TensorProdu …
  -/
  refine g.induction ?_ ?_
    /-
      case refine_1
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      g : Finsupp G k
      f : Fin n → G
      r : k
      ⊢ Eq ((groupCohomology.resolution.diagonalSucc k G n).inv.hom.hom (TensorProdu …
    -/
  · simp only [TensorProduct.zero_tmul, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Group G
      g : Finsupp G k
      f : Fin n → G
      r : k
      ⊢ ∀ (a : G) (b : k) (f_1 : Finsupp G k), Not (Membership.mem f_1.support a) →  …
    -/
  · intro a b x _ _ hx
    simp only [lift_apply, smul_single', map_add, hx, diagonalSucc_inv_single_single,
      TensorProduct.add_tmul, Finsupp.sum_single_index, zero_mul, single_zero]


/-- The `k[G]`-linear isomorphism `k[G] ⊗ₖ k[Gⁿ] ≃ k[Gⁿ⁺¹]`, where the `k[G]`-module structure on
the lefthand side is `TensorProduct.leftModule`, whilst that of the righthand side comes from
`Representation.asModule`. Allows us to use `Algebra.TensorProduct.basis` to get a `k[G]`-basis
of the righthand side. -/
def ofMulActionBasisAux :
    MonoidAlgebra k G ⊗[k] ((Fin n → G) →₀ k) ≃ₗ[MonoidAlgebra k G]
      (ofMulAction k G (Fin (n + 1) → G)).asModule :=
  { (Rep.equivalenceModuleMonoidAlgebra.1.mapIso (diagonalSucc k G n).symm).toLinearEquiv with
    map_smul' := fun r x => by
      /-
        k G : Type u
        inst✝¹ : CommRing k
        n : Nat
        inst✝ : Group G
        r : MonoidAlgebra k G
        x : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
        ⊢ Eq ((↑__src✝).toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id (MonoidAlge …
      -/
      rw [RingHom.id_apply, LinearEquiv.toFun_eq_coe, ← LinearEquiv.map_smul]
      /-
        k G : Type u
        inst✝¹ : CommRing k
        n : Nat
        inst✝ : Group G
        r : MonoidAlgebra k G
        x : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
        ⊢ Eq (__src✝ (HSMul.hSMul r x)) (__src✝ (HSMul.hSMul r x))
      -/
      congr 1
      /- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
      refine' x.induction_on _ (fun x y => _) fun y z hy hz => _
      · simp only [smul_zero]
      · simp only [TensorProduct.smul_tmul']
        show (r * x) ⊗ₜ y = _
        rw [← ofMulAction_self_smul_eq_mul, smul_tprod_one_asModule]
      · rw [smul_add, hz, hy, smul_add] -/
      show _ = Representation.asAlgebraHom (tensorObj (Rep.leftRegular k G)
        (Rep.trivial k G ((Fin n → G) →₀ k))).ρ r _
      /-
        case h.e_6.h
        k G : Type u
        inst✝¹ : CommRing k
        n : Nat
        inst✝ : Group G
        r : MonoidAlgebra k G
        x : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
        ⊢ Eq (HSMul.hSMul r x) (((CategoryTheory.MonoidalCategoryStruct.tensorObj (Rep …
      -/
      refine x.induction_on ?_ (fun x y => ?_) fun y z hy hz => ?_
        /-
          case h.e_6.h.refine_1
          k G : Type u
          inst✝¹ : CommRing k
          n : Nat
          inst✝ : Group G
          r : MonoidAlgebra k G
          x : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
          ⊢ Eq (HSMul.hSMul r 0) (((CategoryTheory.MonoidalCategoryStruct.tensorObj (Rep …
        -/
      · rw [smul_zero, map_zero]
        /-
          🎉 no goals
        -/
        /-
          case h.e_6.h.refine_2
          k G : Type u
          inst✝¹ : CommRing k
          n : Nat
          inst✝ : Group G
          r : MonoidAlgebra k G
          x✝ : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
          x : MonoidAlgebra k G
          y : Finsupp (Fin n → G) k
          ⊢ Eq (HSMul.hSMul r (TensorProduct.tmul k x y)) (((CategoryTheory.MonoidalCate …
        -/
      · rw [TensorProduct.smul_tmul', smul_eq_mul, ← ofMulAction_self_smul_eq_mul]
        /-
          case h.e_6.h.refine_2
          k G : Type u
          inst✝¹ : CommRing k
          n : Nat
          inst✝ : Group G
          r : MonoidAlgebra k G
          x✝ : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
          x : MonoidAlgebra k G
          y : Finsupp (Fin n → G) k
          ⊢ Eq (TensorProduct.tmul k (HSMul.hSMul r x) y) (((CategoryTheory.MonoidalCate …
        -/
        exact (smul_tprod_one_asModule (Representation.ofMulAction k G G) r x y).symm
        /-
          🎉 no goals
        -/
        /-
          case h.e_6.h.refine_3
          k G : Type u
          inst✝¹ : CommRing k
          n : Nat
          inst✝ : Group G
          r : MonoidAlgebra k G
          x y z : TensorProduct k (MonoidAlgebra k G) (Finsupp (Fin n → G) k)
          hy : Eq (HSMul.hSMul r y) (((CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
          hz : Eq (HSMul.hSMul r z) (((CategoryTheory.MonoidalCategoryStruct.tensorObj ( …
          ⊢ Eq (HSMul.hSMul r (HAdd.hAdd y z)) (((CategoryTheory.MonoidalCategoryStruct. …
        -/
      · rw [smul_add, hz, hy, map_add] }
        /-
          🎉 no goals
        -/


/-- A `k[G]`-basis of `k[Gⁿ⁺¹]`, coming from the `k[G]`-linear isomorphism
`k[G] ⊗ₖ k[Gⁿ] ≃ k[Gⁿ⁺¹].` -/
def ofMulActionBasis :
    Basis (Fin n → G) (MonoidAlgebra k G) (ofMulAction k G (Fin (n + 1) → G)).asModule :=
  Basis.map
    (Algebra.TensorProduct.basis (MonoidAlgebra k G)
      (Finsupp.basisSingleOne : Basis (Fin n → G) k ((Fin n → G) →₀ k)))
    (ofMulActionBasisAux k G n)


theorem ofMulAction_free :
    Module.Free (MonoidAlgebra k G) (ofMulAction k G (Fin (n + 1) → G)).asModule :=
  Module.Free.of_basis (ofMulActionBasis k G n)


/-- Given a `k`-linear `G`-representation `A`, the set of representation morphisms
`Hom(k[Gⁿ⁺¹], A)` is `k`-linearly isomorphic to the set of functions `Gⁿ → A`. -/
noncomputable def diagonalHomEquiv :
    (Rep.ofMulAction k G (Fin (n + 1) → G) ⟶ A) ≃ₗ[k] (Fin n → G) → A :=
  Linear.homCongr k
        ((diagonalSucc k G n).trans ((Representation.ofMulAction k G G).repOfTprodIso 1))
        (Iso.refl _) ≪≫ₗ
      (Rep.MonoidalClosed.linearHomEquivComm _ _ _ ≪≫ₗ Rep.leftRegularHomEquiv _) ≪≫ₗ
    (Finsupp.llift A k k (Fin n → G)).symm


/-- Given a `k`-linear `G`-representation `A`, `diagonalHomEquiv` is a `k`-linear isomorphism of
the set of representation morphisms `Hom(k[Gⁿ⁺¹], A)` with `Fun(Gⁿ, A)`. This lemma says that this
sends a morphism of representations `f : k[Gⁿ⁺¹] ⟶ A` to the function
`(g₁, ..., gₙ) ↦ f(1, g₁, g₁g₂, ..., g₁g₂...gₙ).` -/
theorem diagonalHomEquiv_apply (f : Rep.ofMulAction k G (Fin (n + 1) → G) ⟶ A) (x : Fin n → G) :
    diagonalHomEquiv n A f x = f.hom (Finsupp.single (Fin.partialProd x) 1) := by
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
  unfold diagonalHomEquiv
  simpa only [LinearEquiv.trans_apply, Rep.leftRegularHomEquiv_apply,
    MonoidalClosed.linearHomEquivComm_hom, Finsupp.llift_symm_apply, TensorProduct.curry_apply,
    Linear.homCongr_apply, Iso.refl_hom, Iso.trans_inv, Action.comp_hom, ModuleCat.comp_def,
    LinearMap.comp_apply, Representation.repOfTprodIso_inv_apply,
    diagonalSucc_inv_single_single (1 : G) x, one_smul, one_mul] -/
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : Quiver.Hom (Rep.ofMulAction k G (Fin (HAdd.hAdd n 1) → G)) A
    x : Fin n → G
    ⊢ Eq ((Rep.diagonalHomEquiv n A) f x) (f.hom.hom (Finsupp.single (Fin.partialP …
  -/
  change f.hom ((diagonalSucc k G n).inv.hom (Finsupp.single 1 1 ⊗ₜ[k] Finsupp.single x 1)) = _
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : Quiver.Hom (Rep.ofMulAction k G (Fin (HAdd.hAdd n 1) → G)) A
    x : Fin n → G
    ⊢ Eq (f.hom.hom ((groupCohomology.resolution.diagonalSucc k G n).inv.hom.hom ( …
  -/
  rw [diagonalSucc_inv_single_single, one_smul, one_mul]
  /-
    🎉 no goals
  -/


/-- Given a `k`-linear `G`-representation `A`, `diagonalHomEquiv` is a `k`-linear isomorphism of
the set of representation morphisms `Hom(k[Gⁿ⁺¹], A)` with `Fun(Gⁿ, A)`. This lemma says that the
inverse map sends a function `f : Gⁿ → A` to the representation morphism sending
`(g₀, ... gₙ) ↦ ρ(g₀)(f(g₀⁻¹g₁, g₁⁻¹g₂, ..., gₙ₋₁⁻¹gₙ))`, where `ρ` is the representation attached
to `A`. -/
theorem diagonalHomEquiv_symm_apply (f : (Fin n → G) → A) (x : Fin (n + 1) → G) :
    ((diagonalHomEquiv n A).symm f).hom (Finsupp.single x 1) =
      A.ρ (x 0) (f fun i : Fin n => (x (Fin.castSucc i))⁻¹ * x i.succ) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    x : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq (((Rep.diagonalHomEquiv n A).symm f).hom.hom (Finsupp.single x 1)) ((A.ρ  …
  -/
  unfold diagonalHomEquiv
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
  simp only [LinearEquiv.trans_symm, LinearEquiv.symm_symm, LinearEquiv.trans_apply,
    Rep.leftRegularHomEquiv_symm_apply, Linear.homCongr_symm_apply, Action.comp_hom, Iso.refl_inv,
    Category.comp_id, Rep.MonoidalClosed.linearHomEquivComm_symm_hom, Iso.trans_hom,
    ModuleCat.comp_def, LinearMap.comp_apply, Representation.repOfTprodIso_apply,
    diagonalSucc_hom_single x (1 : k), TensorProduct.uncurry_apply, Rep.leftRegularHom_hom,
    Finsupp.lift_apply, ihom_obj_ρ_def, Rep.ihom_obj_ρ_apply, Finsupp.sum_single_index, zero_smul,
    one_smul, Rep.of_ρ, Rep.Action_ρ_eq_ρ, Rep.trivial_def (x 0)⁻¹, Finsupp.llift_apply A k k] -/
  simp only [LinearEquiv.trans_symm, LinearEquiv.symm_symm, LinearEquiv.trans_apply,
    leftRegularHomEquiv_symm_apply, Linear.homCongr_symm_apply, Iso.trans_hom, Iso.refl_inv,
    Category.comp_id, Action.comp_hom, MonoidalClosed.linearHomEquivComm_symm_hom,
    ModuleCat.hom_comp, LinearMap.comp_apply]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    x : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq (((TensorProduct.uncurry k (CoeSort.coe (Rep.of (Representation.ofMulActi …
  -/
  erw [diagonalSucc_hom_single]
  -- The prototype linter that checks if `erw` could be replaced with `rw` would time out
  -- if it replaces the next `erw`s with `rw`s. So we focus down on the relevant part.
  conv_lhs =>
    erw [TensorProduct.uncurry_apply, Finsupp.lift_apply, Finsupp.sum_single_index]
    · simp only [one_smul]
      erw [Representation.linHom_apply]
      simp only [LinearMap.comp_apply, MonoidHom.one_apply, LinearMap.one_apply]
      erw [Finsupp.llift_apply]
      rw [Finsupp.lift_apply]
      erw [Finsupp.sum_single_index]
      · rw [one_smul]
      · rw [zero_smul]
    · rw [zero_smul]


/-- Auxiliary lemma for defining group cohomology, used to show that the isomorphism
`diagonalHomEquiv` commutes with the differentials in two complexes which compute
group cohomology. -/
theorem diagonalHomEquiv_symm_partialProd_succ (f : (Fin n → G) → A) (g : Fin (n + 1) → G)
    (a : Fin (n + 1)) :
    ((diagonalHomEquiv n A).symm f).hom (Finsupp.single (Fin.partialProd g ∘ a.succ.succAbove) 1)
      = f (Fin.contractNth a (· * ·) g) := by
  simp only [diagonalHomEquiv_symm_apply, Function.comp_apply, Fin.succ_succAbove_zero,
    Fin.partialProd_zero, map_one, Fin.succ_succAbove_succ, LinearMap.one_apply,
    Fin.partialProd_succ]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    g : Fin (HAdd.hAdd n 1) → G
    a : Fin (HAdd.hAdd n 1)
    ⊢ Eq (f fun i => HMul.hMul (Inv.inv (Fin.partialProd g (a.succ.succAbove i.cas …
  -/
  congr
  /-
    case e_a
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    g : Fin (HAdd.hAdd n 1) → G
    a : Fin (HAdd.hAdd n 1)
    ⊢ Eq (fun i => HMul.hMul (Inv.inv (Fin.partialProd g (a.succ.succAbove i.castS …
  -/
  ext
  /-
    case e_a.h
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Group G
    A : Rep k G
    f : (Fin n → G) → CoeSort.coe A
    g : Fin (HAdd.hAdd n 1) → G
    a : Fin (HAdd.hAdd n 1)
    x✝ : Fin n
    ⊢ Eq (HMul.hMul (Inv.inv (Fin.partialProd g (a.succ.succAbove x✝.castSucc))) ( …
  -/
  rw [← Fin.partialProd_succ, Fin.inv_partialProd_mul_eq_contractNth]
  /-
    🎉 no goals
  -/


/-- The simplicial `G`-set sending `[n]` to `Gⁿ⁺¹` equipped with the diagonal action of `G`. -/
def classifyingSpaceUniversalCover [Monoid G] :
    SimplicialObject (Action (Type u) <| MonCat.of G) where
  obj n := Action.ofMulAction G (Fin (n.unop.len + 1) → G)
  map f :=
    { hom := fun x => x ∘ f.unop.toOrderHom
      comm := fun _ => rfl }
  map_id _ := rfl
  map_comp _ _ := rfl


/-- When the category is `G`-Set, `cechNerveTerminalFrom` of `G` with the left regular action is
isomorphic to `EG`, the universal cover of the classifying space of `G` as a simplicial `G`-set. -/
def cechNerveTerminalFromIso :
    cechNerveTerminalFrom (Action.ofMulAction G G) ≅ classifyingSpaceUniversalCover G :=
  NatIso.ofComponents (fun _ => limit.isoLimitCone (Action.ofMulActionLimitCone _ _)) fun f => by
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Monoid G
      X✝ Y✝ : Opposite SimplexCategory
      f : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.cechNerveTerminalFro …
    -/
    refine IsLimit.hom_ext (Action.ofMulActionLimitCone.{u, 0} G fun _ => G).2 fun j => ?_
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Monoid G
      X✝ Y✝ : Opposite SimplexCategory
      f : Quiver.Hom X✝ Y✝
      j : CategoryTheory.Discrete (Fin (HAdd.hAdd (Opposite.unop Y✝).len 1))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    dsimp only [cechNerveTerminalFrom, Pi.lift]
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Monoid G
      X✝ Y✝ : Opposite SimplexCategory
      f : Quiver.Hom X✝ Y✝
      j : CategoryTheory.Discrete (Fin (HAdd.hAdd (Opposite.unop Y✝).len 1))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Category.assoc, limit.isoLimitCone_hom_π, limit.lift_π, Category.assoc]
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Monoid G
      X✝ Y✝ : Opposite SimplexCategory
      f : Quiver.Hom X✝ Y✝
      j : CategoryTheory.Discrete (Fin (HAdd.hAdd (Opposite.unop Y✝).len 1))
      ⊢ Eq ((CategoryTheory.Limits.Fan.mk (CategoryTheory.Limits.piObj fun x => Acti …
    -/
    exact (limit.isoLimitCone_hom_π _ _).symm
    /-
      🎉 no goals
    -/


/-- As a simplicial set, `cechNerveTerminalFrom` of a monoid `G` is isomorphic to the universal
cover of the classifying space of `G` as a simplicial set. -/
def cechNerveTerminalFromIsoCompForget :
    cechNerveTerminalFrom G ≅ classifyingSpaceUniversalCover G ⋙ forget _ :=
  NatIso.ofComponents (fun _ => Types.productIso _) fun _ =>
    Matrix.ext fun _ _ => Types.Limit.lift_π_apply (Discrete.functor fun _ ↦ G) _ _ _


/-- The universal cover of the classifying space of `G` as a simplicial set, augmented by the map
from `Fin 1 → G` to the terminal object in `Type u`. -/
def compForgetAugmented : SimplicialObject.Augmented (Type u) :=
  SimplicialObject.augment (classifyingSpaceUniversalCover G ⋙ forget _) (terminal _)
    (terminal.from _) fun _ _ _ => Subsingleton.elim _ _


/-- The augmented Čech nerve of the map from `Fin 1 → G` to the terminal object in `Type u` has an
extra degeneracy. -/
def extraDegeneracyAugmentedCechNerve :
    ExtraDegeneracy (Arrow.mk <| terminal.from G).augmentedCechNerve :=
  AugmentedCechNerve.extraDegeneracy (Arrow.mk <| terminal.from G)
    ⟨fun _ => (1 : G),
      @Subsingleton.elim _ (@Unique.instSubsingleton _ (Limits.uniqueToTerminal _)) _ _⟩


/-- The universal cover of the classifying space of `G` as a simplicial set, augmented by the map
from `Fin 1 → G` to the terminal object in `Type u`, has an extra degeneracy. -/
def extraDegeneracyCompForgetAugmented : ExtraDegeneracy (compForgetAugmented G) := by
  refine
    ExtraDegeneracy.ofIso (?_ : (Arrow.mk <| terminal.from G).augmentedCechNerve ≅ _)
      (extraDegeneracyAugmentedCechNerve G)
  exact
    Comma.isoMk (CechNerveTerminalFrom.iso G ≪≫ cechNerveTerminalFromIsoCompForget G)
      (Iso.refl _) (by ext : 1; exact IsTerminal.hom_ext terminalIsTerminal _ _)


/-- The free functor `Type u ⥤ ModuleCat.{u} k` applied to the universal cover of the classifying
space of `G` as a simplicial set, augmented by the map from `Fin 1 → G` to the terminal object
in `Type u`. -/
def compForgetAugmented.toModule : SimplicialObject.Augmented (ModuleCat.{u} k) :=
  ((SimplicialObject.Augmented.whiskering _ _).obj (ModuleCat.free k)).obj (compForgetAugmented G)


/-- If we augment the universal cover of the classifying space of `G` as a simplicial set by the
map from `Fin 1 → G` to the terminal object in `Type u`, then apply the free functor
`Type u ⥤ ModuleCat.{u} k`, the resulting augmented simplicial `k`-module has an extra
degeneracy. -/
def extraDegeneracyCompForgetAugmentedToModule :
    ExtraDegeneracy (compForgetAugmented.toModule k G) :=
  ExtraDegeneracy.map (extraDegeneracyCompForgetAugmented G) (ModuleCat.free k)


/-- The standard resolution of `k` as a trivial representation, defined as the alternating
face map complex of a simplicial `k`-linear `G`-representation. -/
def groupCohomology.resolution [Monoid G] :=
  (AlgebraicTopology.alternatingFaceMapComplex (Rep k G)).obj
    (classifyingSpaceUniversalCover G ⋙ Rep.linearization k G)


/-- The `k`-linear map underlying the differential in the standard resolution of `k` as a trivial
`k`-linear `G`-representation. It sends `(g₀, ..., gₙ) ↦ ∑ (-1)ⁱ • (g₀, ..., ĝᵢ, ..., gₙ)`. -/
def d (G : Type u) (n : ℕ) : ((Fin (n + 1) → G) →₀ k) →ₗ[k] (Fin n → G) →₀ k :=
  Finsupp.lift ((Fin n → G) →₀ k) k (Fin (n + 1) → G) fun g =>
    (@Finset.univ (Fin (n + 1)) _).sum fun p =>
      Finsupp.single (g ∘ p.succAbove) ((-1 : k) ^ (p : ℕ))


@[simp]
theorem d_of {G : Type u} {n : ℕ} (c : Fin (n + 1) → G) :
    d k G n (Finsupp.single c 1) =
      Finset.univ.sum fun p : Fin (n + 1) =>
        Finsupp.single (c ∘ p.succAbove) ((-1 : k) ^ (p : ℕ)) := by
  /-
    k : Type u
    inst✝ : CommRing k
    G : Type u
    n : Nat
    c : Fin (HAdd.hAdd n 1) → G
    ⊢ Eq ((groupCohomology.resolution.d k G n) (Finsupp.single c 1)) (Finset.univ. …
  -/
  simp [d]
  /-
    🎉 no goals
  -/


/-- The `n`th object of the standard resolution of `k` is definitionally isomorphic to `k[Gⁿ⁺¹]`
equipped with the representation induced by the diagonal action of `G`. -/
def xIso (n : ℕ) : (groupCohomology.resolution k G).X n ≅ Rep.ofMulAction k G (Fin (n + 1) → G) :=
  Iso.refl _


instance x_projective (G : Type u) [Group G] (n : ℕ) :
    Projective ((groupCohomology.resolution k G).X n) :=
  Rep.equivalenceModuleMonoidAlgebra.toAdjunction.projective_of_map_projective _ <|
    @ModuleCat.projective_of_free.{u} _ _
      (ModuleCat.of (MonoidAlgebra k G) (Representation.ofMulAction k G (Fin (n + 1) → G)).asModule)
      _ (ofMulActionBasis k G n)


/-- Simpler expression for the differential in the standard resolution of `k` as a
`G`-representation. It sends `(g₀, ..., gₙ₊₁) ↦ ∑ (-1)ⁱ • (g₀, ..., ĝᵢ, ..., gₙ₊₁)`. -/
theorem d_eq (n : ℕ) : ((groupCohomology.resolution k G).d (n + 1) n).hom =
    ModuleCat.ofHom (d k G (n + 1)) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    ⊢ Eq ((groupCohomology.resolution k G).d (HAdd.hAdd n 1) n).hom (ModuleCat.ofH …
  -/
  ext : 1
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    ⊢ Eq ((groupCohomology.resolution k G).d (HAdd.hAdd n 1) n).hom.hom (ModuleCat …
  -/
  refine Finsupp.lhom_ext' fun x => LinearMap.ext_ring ?_
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    ⊢ Eq ((((groupCohomology.resolution k G).d (HAdd.hAdd n 1) n).hom.hom.comp (Fi …
  -/
  dsimp [groupCohomology.resolution]
/- Porting note (https://github.com/leanprover-community/mathlib4/issues/11039): broken proof was
  simpa [← @intCast_smul k, simplicial_object.δ] -/
  simp_rw [alternatingFaceMapComplex_obj_d, AlternatingFaceMapComplex.objD, SimplicialObject.δ,
    Functor.comp_map, ← Int.cast_smul_eq_zsmul k ((-1) ^ _ : ℤ), Int.cast_pow, Int.cast_neg,
    Int.cast_one, Action.sum_hom, Action.smul_hom, Rep.linearization_map_hom]
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    ⊢ Eq ((Finset.univ.sum fun x => HSMul.hSMul (HPow.hPow (-1) ↑x) (ModuleCat.ofH …
  -/
  rw [ModuleCat.hom_sum, LinearMap.coeFn_sum, Fintype.sum_apply]
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    ⊢ Eq (Finset.univ.sum fun c => (HSMul.hSMul (HPow.hPow (-1) ↑c) (ModuleCat.ofH …
  -/
  erw [d_of (k := k) x]
/- Porting note: want to rewrite `LinearMap.smul_apply` but simp/simp_rw won't do it; I need erw,
so using Finset.sum_congr to get rid of the binder -/
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    ⊢ Eq (Finset.univ.sum fun c => (HSMul.hSMul (HPow.hPow (-1) ↑c) (ModuleCat.ofH …
  -/
  refine Finset.sum_congr rfl fun _ _ => ?_
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    x✝¹ : Fin (HAdd.hAdd n 2)
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq ((HSMul.hSMul (HPow.hPow (-1) ↑x✝¹) (ModuleCat.ofHom (Finsupp.lmapDomain  …
  -/
  simp only [ModuleCat.hom_smul, SimplexCategory.len_mk]
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    x✝¹ : Fin (HAdd.hAdd n 2)
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq ((HSMul.hSMul (HPow.hPow (-1) ↑x✝¹) (Finsupp.lmapDomain k k ((classifying …
  -/
  erw [LinearMap.smul_apply]
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    x✝¹ : Fin (HAdd.hAdd n 2)
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) ↑x✝¹) ((Finsupp.lmapDomain k k ((classifying …
  -/
  rw [Finsupp.lmapDomain_apply, Finsupp.mapDomain_single, Finsupp.smul_single', mul_one]
  /-
    case hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    n : Nat
    x : ((classifyingSpaceUniversalCover G).obj { unop := SimplexCategory.mk (HAdd …
    x✝¹ : Fin (HAdd.hAdd n 2)
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq (Finsupp.single (((classifyingSpaceUniversalCover G).map (SimplexCategory …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The standard resolution of `k` as a trivial representation as a complex of `k`-modules. -/
def forget₂ToModuleCat :=
  ((forget₂ (Rep k G) (ModuleCat.{u} k)).mapHomologicalComplex _).obj
    (groupCohomology.resolution k G)


/-- If we apply the free functor `Type u ⥤ ModuleCat.{u} k` to the universal cover of the
classifying space of `G` as a simplicial set, then take the alternating face map complex, the result
is isomorphic to the standard resolution of the trivial `G`-representation `k` as a complex of
`k`-modules. -/
def compForgetAugmentedIso :
    AlternatingFaceMapComplex.obj
        (SimplicialObject.Augmented.drop.obj (compForgetAugmented.toModule k G)) ≅
      groupCohomology.resolution.forget₂ToModuleCat k G :=
  eqToIso
    (Functor.congr_obj (map_alternatingFaceMapComplex (forget₂ (Rep k G) (ModuleCat.{u} k))).symm
      (classifyingSpaceUniversalCover G ⋙ Rep.linearization k G))


/-- As a complex of `k`-modules, the standard resolution of the trivial `G`-representation `k` is
homotopy equivalent to the complex which is `k` at 0 and 0 elsewhere. -/
def forget₂ToModuleCatHomotopyEquiv :
    HomotopyEquiv (groupCohomology.resolution.forget₂ToModuleCat k G)
      ((ChainComplex.single₀ (ModuleCat k)).obj ((forget₂ (Rep k G) _).obj <| Rep.trivial k G k)) :=
  (HomotopyEquiv.ofIso (compForgetAugmentedIso k G).symm).trans <|
    (SimplicialObject.Augmented.ExtraDegeneracy.homotopyEquiv
          (extraDegeneracyCompForgetAugmentedToModule k G)).trans
      (HomotopyEquiv.ofIso <|
        (ChainComplex.single₀ (ModuleCat.{u} k)).mapIso
          (@Finsupp.LinearEquiv.finsuppUnique k k _ _ _ (⊤_ Type u)
              Types.terminalIso.toEquiv.unique).toModuleIso)


/-- The hom of `k`-linear `G`-representations `k[G¹] → k` sending `∑ nᵢgᵢ ↦ ∑ nᵢ`. -/
def ε : Rep.ofMulAction k G (Fin 1 → G) ⟶ Rep.trivial k G k where
  hom := ModuleCat.ofHom <| Finsupp.linearCombination _ fun _ => (1 : k)
  comm g := ModuleCat.hom_ext <| Finsupp.lhom_ext' fun _ => LinearMap.ext_ring (by
    show
      Finsupp.linearCombination k (fun _ => (1 : k)) (Finsupp.mapDomain _ (Finsupp.single _ _)) =
        Finsupp.linearCombination k (fun _ => (1 : k)) (Finsupp.single _ _)
    /-
      k G : Type u
      inst✝¹ : CommRing k
      n : Nat
      inst✝ : Monoid G
      g : ↑(MonCat.of G)
      x✝ : Fin 1 → G
      ⊢ Eq ((Finsupp.linearCombination k fun x => 1) (Finsupp.mapDomain (fun x => HS …
    -/
    simp only [Finsupp.mapDomain_single, Finsupp.linearCombination_single])
    /-
      🎉 no goals
    -/


/-- The homotopy equivalence of complexes of `k`-modules between the standard resolution of `k` as
a trivial `G`-representation, and the complex which is `k` at 0 and 0 everywhere else, acts as
`∑ nᵢgᵢ ↦ ∑ nᵢ : k[G¹] → k` at 0. -/
theorem forget₂ToModuleCatHomotopyEquiv_f_0_eq :
    (forget₂ToModuleCatHomotopyEquiv k G).1.f 0 = (forget₂ (Rep k G) _).map (ε k G) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq ((groupCohomology.resolution.forget₂ToModuleCatHomotopyEquiv k G).hom.f 0 …
  -/
  show (HomotopyEquiv.hom _ ≫ HomotopyEquiv.hom _ ≫ HomotopyEquiv.hom _).f 0 = _
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (HomotopyEquiv.ofIso (groupCohomolog …
  -/
  simp only [HomologicalComplex.comp_f]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyEquiv.ofIso (groupCohomolog …
  -/
  dsimp
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyEquiv.ofIso (groupCohomolog …
  -/
  convert Category.id_comp (X := (forget₂ToModuleCat k G).X 0) _
    /-
      case h.e'_2.h.h.e'_6.h
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      e_4✝ : Eq ((classifyingSpaceUniversalCover.compForgetAugmented.toModule k G).l …
      ⊢ Eq ((HomotopyEquiv.ofIso (groupCohomology.resolution.compForgetAugmentedIso  …
    -/
  · dsimp only [HomotopyEquiv.ofIso, compForgetAugmentedIso]
    /-
      case h.e'_2.h.h.e'_6.h
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      e_4✝ : Eq ((classifyingSpaceUniversalCover.compForgetAugmented.toModule k G).l …
      ⊢ Eq ((CategoryTheory.eqToIso ⋯).symm.hom.f 0) (CategoryTheory.CategoryStruct. …
    -/
    simp only [Iso.symm_hom, eqToIso.inv, HomologicalComplex.eqToHom_f, eqToHom_refl]
    /-
      🎉 no goals
    -/
  /-
    case h.e'_3.h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq ((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).map (groupCohomology.re …
  -/
  ext : 1
  /-
    case h.e'_3.h.hf
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq ((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).map (groupCohomology.re …
  -/
  trans (linearCombination _ fun _ => (1 : k)).comp ((ModuleCat.free k).map (terminal.from _)).hom
  · erw [Finsupp.lmapDomain_linearCombination (α := Fin 1 → G) (R := k) (α' := ⊤_ Type u)
        (v := fun _ => (1 : k)) (v' := fun _ => (1 : k))
        (terminal.from
          ((classifyingSpaceUniversalCover G).obj (Opposite.op (SimplexCategory.mk 0))).V)
        LinearMap.id fun i => rfl,
      LinearMap.id_comp]
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      ⊢ Eq ((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).map (groupCohomology.re …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      k G : Type u
      inst✝¹ : CommRing k
      inst✝ : Monoid G
      ⊢ Eq ((Finsupp.linearCombination k fun x => 1).comp ((ModuleCat.free k).map (C …
    -/
  · congr
      /-
        case e_f
        k G : Type u
        inst✝¹ : CommRing k
        inst✝ : Monoid G
        ⊢ Eq (Finsupp.linearCombination k fun x => 1) ((HomotopyEquiv.ofIso ((ChainCom …
      -/
    · ext x
      dsimp (config := { unfoldPartialApp := true }) [HomotopyEquiv.ofIso,
        Finsupp.LinearEquiv.finsuppUnique]
      /-
        case e_f.h.h
        k G : Type u
        inst✝¹ : CommRing k
        inst✝ : Monoid G
        x : CategoryTheory.Limits.terminal (Type u)
        ⊢ Eq ((Finsupp.linearCombination k fun x => 1) (Finsupp.single x 1)) ((((Chain …
      -/
      rw [@Unique.eq_default _ Types.terminalIso.toEquiv.unique x]
      /-
        case e_f.h.h
        k G : Type u
        inst✝¹ : CommRing k
        inst✝ : Monoid G
        x : CategoryTheory.Limits.terminal (Type u)
        ⊢ Eq ((Finsupp.linearCombination k fun x => 1) (Finsupp.single Inhabited.defau …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case e_g.e_self.e_a
        k G : Type u
        inst✝¹ : CommRing k
        inst✝ : Monoid G
        ⊢ Eq (CategoryTheory.Limits.terminal.from (Fin 1 → G)) ((classifyingSpaceUnive …
      -/
    · exact @Subsingleton.elim _ (@Unique.instSubsingleton _ (Limits.uniqueToTerminal _)) _ _
      /-
        🎉 no goals
      -/


theorem d_comp_ε : (groupCohomology.resolution k G).d 1 0 ≫ ε k G = 0 := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((groupCohomology.resolution k G).d 1 …
  -/
  ext : 1
  /-
    case h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((groupCohomology.resolution k G).d 1 …
  -/
  refine ModuleCat.hom_ext <| LinearMap.ext fun x => ?_
  have : (forget₂ToModuleCat k G).d 1 0
      ≫ (forget₂ (Rep k G) (ModuleCat.{u} k)).map (ε k G) = 0 := by
    rw [← forget₂ToModuleCatHomotopyEquiv_f_0_eq,
      ← (forget₂ToModuleCatHomotopyEquiv k G).1.2 1 0 rfl]
    exact comp_zero
  /-
    case h
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    x : ↑((groupCohomology.resolution k G).X 1).V
    this : Eq (CategoryTheory.CategoryStruct.comp ((groupCohomology.resolution.for …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((groupCohomology.resolution k G).d  …
  -/
  exact LinearMap.ext_iff.1 (ModuleCat.hom_ext_iff.mp this) _
  /-
    🎉 no goals
  -/


/-- The chain map from the standard resolution of `k` to `k[0]` given by `∑ nᵢgᵢ ↦ ∑ nᵢ` in
degree zero. -/
def εToSingle₀ :
    groupCohomology.resolution k G ⟶ (ChainComplex.single₀ _).obj (Rep.trivial k G k) :=
  ((groupCohomology.resolution k G).toSingle₀Equiv _).symm ⟨ε k G, d_comp_ε k G⟩


theorem εToSingle₀_comp_eq :
    ((forget₂ _ (ModuleCat.{u} k)).mapHomologicalComplex _).map (εToSingle₀ k G) ≫
        (HomologicalComplex.singleMapHomologicalComplex _ _ _).hom.app _ =
      (forget₂ToModuleCatHomotopyEquiv k G).hom := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ (Rep k G) ( …
  -/
  dsimp
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ (Rep k G) ( …
  -/
  ext1
  /-
    case hfg
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ (Rep k G)  …
  -/
  dsimp
  /-
    case hfg
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.forget₂ (Rep k G) (M …
  -/
  simpa using (forget₂ToModuleCatHomotopyEquiv_f_0_eq k G).symm
  /-
    🎉 no goals
  -/


theorem quasiIso_forget₂_εToSingle₀ :
    QuasiIso (((forget₂ _ (ModuleCat.{u} k)).mapHomologicalComplex _).map (εToSingle₀ k G)) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    ⊢ QuasiIso (((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).mapHomologicalCo …
  -/
  have h : QuasiIso (forget₂ToModuleCatHomotopyEquiv k G).hom := inferInstance
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    h : QuasiIso (groupCohomology.resolution.forget₂ToModuleCatHomotopyEquiv k G). …
    ⊢ QuasiIso (((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).mapHomologicalCo …
  -/
  rw [← εToSingle₀_comp_eq k G] at h
  /-
    k G : Type u
    inst✝¹ : CommRing k
    inst✝ : Monoid G
    h : QuasiIso (CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ (Re …
    ⊢ QuasiIso (((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).mapHomologicalCo …
  -/
  exact quasiIso_of_comp_right (hφφ' := h)
  /-
    🎉 no goals
  -/


instance : QuasiIso (εToSingle₀ k G) := by
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Monoid G
    ⊢ QuasiIso (groupCohomology.resolution.εToSingle₀ k G)
  -/
  rw [← HomologicalComplex.quasiIso_map_iff_of_preservesHomology _ (forget₂ _ (ModuleCat.{u} k))]
  /-
    k G : Type u
    inst✝¹ : CommRing k
    n : Nat
    inst✝ : Monoid G
    ⊢ QuasiIso (((CategoryTheory.forget₂ (Rep k G) (ModuleCat k)).mapHomologicalCo …
  -/
  apply quasiIso_forget₂_εToSingle₀
  /-
    🎉 no goals
  -/


/-- The standard projective resolution of `k` as a trivial `k`-linear `G`-representation. -/
def groupCohomology.projectiveResolution : ProjectiveResolution (Rep.trivial k G k) where
  π := εToSingle₀ k G


instance : EnoughProjectives (Rep k G) :=
  Rep.equivalenceModuleMonoidAlgebra.enoughProjectives_iff.2
    ModuleCat.moduleCat_enoughProjectives.{u}


/-- Given a `k`-linear `G`-representation `V`, `Extⁿ(k, V)` (where `k` is a trivial `k`-linear
`G`-representation) is isomorphic to the `n`th cohomology group of `Hom(P, V)`, where `P` is the
standard resolution of `k` called `groupCohomology.resolution k G`. -/
def groupCohomology.extIso (V : Rep k G) (n : ℕ) :
    ((Ext k (Rep k G) n).obj (Opposite.op <| Rep.trivial k G k)).obj V ≅
      ((groupCohomology.resolution k G).linearYonedaObj k V).homology n :=
  (groupCohomology.projectiveResolution k G).isoExt n V

