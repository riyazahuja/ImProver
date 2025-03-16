/-- The exterior algebra of an `R`-module `M`.
-/
abbrev ExteriorAlgebra :=
  CliffordAlgebra (0 : QuadraticForm R M)


/-- The canonical linear map `M →ₗ[R] ExteriorAlgebra R M`.
-/
abbrev ι : M →ₗ[R] ExteriorAlgebra R M :=
  CliffordAlgebra.ι _


/-- Definition of the `n`th exterior power of a `R`-module `N`. We introduce the notation
`⋀[R]^n M` for `exteriorPower R n M`. -/
abbrev exteriorPower : Submodule R (ExteriorAlgebra R M) :=
  LinearMap.range (ι R : M →ₗ[R] ExteriorAlgebra R M) ^ n


@[inherit_doc exteriorPower]
notation:max "⋀[" R "]^" n:arg => exteriorPower R n


/-- As well as being linear, `ι m` squares to zero. -/
theorem ι_sq_zero (m : M) : ι R m * ι R m = 0 :=
  (CliffordAlgebra.ι_sq_scalar _ m).trans <| map_zero _


theorem comp_ι_sq_zero (g : ExteriorAlgebra R M →ₐ[R] A) (m : M) : g (ι R m) * g (ι R m) = 0 := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (ExteriorAlgebra R M) A
    m : M
    ⊢ Eq (HMul.hMul (g ((ExteriorAlgebra.ι R) m)) (g ((ExteriorAlgebra.ι R) m))) 0
  -/
  rw [← map_mul, ι_sq_zero, map_zero]
  /-
    🎉 no goals
  -/


/-- Given a linear map `f : M →ₗ[R] A` into an `R`-algebra `A`, which satisfies the condition:
`cond : ∀ m : M, f m * f m = 0`, this is the canonical lift of `f` to a morphism of `R`-algebras
from `ExteriorAlgebra R M` to `A`.
-/
@[simps! symm_apply]
def lift : { f : M →ₗ[R] A // ∀ m, f m * f m = 0 } ≃ (ExteriorAlgebra R M →ₐ[R] A) :=
                                                       /-
                                                         R : Type u1
                                                         inst✝⁴ : CommRing R
                                                         M : Type u2
                                                         inst✝³ : AddCommGroup M
                                                         inst✝² : Module R M
                                                         A : Type u_1
                                                         inst✝¹ : Semiring A
                                                         inst✝ : Algebra R A
                                                         ⊢ ∀ (a : LinearMap (RingHom.id R) M A), Iff (∀ (m : M), Eq (HMul.hMul (a m) (a …
                                                       -/
  Equiv.trans (Equiv.subtypeEquiv (Equiv.refl _) <| by simp) <| CliffordAlgebra.lift _
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem ι_comp_lift (f : M →ₗ[R] A) (cond : ∀ m, f m * f m = 0) :
    (lift R ⟨f, cond⟩).toLinearMap.comp (ι R) = f :=
  CliffordAlgebra.ι_comp_lift f _


@[simp]
theorem lift_ι_apply (f : M →ₗ[R] A) (cond : ∀ m, f m * f m = 0) (x) :
    lift R ⟨f, cond⟩ (ι R x) = f x :=
  CliffordAlgebra.lift_ι_apply f _ x


@[simp]
theorem lift_unique (f : M →ₗ[R] A) (cond : ∀ m, f m * f m = 0) (g : ExteriorAlgebra R M →ₐ[R] A) :
    g.toLinearMap.comp (ι R) = f ↔ g = lift R ⟨f, cond⟩ :=
  CliffordAlgebra.lift_unique f _ _


@[simp]
theorem lift_comp_ι (g : ExteriorAlgebra R M →ₐ[R] A) :
    lift R ⟨g.toLinearMap.comp (ι R), comp_ι_sq_zero _⟩ = g :=
  CliffordAlgebra.lift_comp_ι g


/-- See note [partially-applied ext lemmas]. -/
@[ext]
theorem hom_ext {f g : ExteriorAlgebra R M →ₐ[R] A}
    (h : f.toLinearMap.comp (ι R) = g.toLinearMap.comp (ι R)) : f = g :=
  CliffordAlgebra.hom_ext h


/-- If `C` holds for the `algebraMap` of `r : R` into `ExteriorAlgebra R M`, the `ι` of `x : M`,
and is preserved under addition and multiplication, then it holds for all of `ExteriorAlgebra R M`.
-/
@[elab_as_elim]
theorem induction {C : ExteriorAlgebra R M → Prop}
    (algebraMap : ∀ r, C (algebraMap R (ExteriorAlgebra R M) r)) (ι : ∀ x, C (ι R x))
    (mul : ∀ a b, C a → C b → C (a * b)) (add : ∀ a b, C a → C b → C (a + b))
    (a : ExteriorAlgebra R M) : C a :=
  CliffordAlgebra.induction algebraMap ι mul add a


/-- The left-inverse of `algebraMap`. -/
def algebraMapInv : ExteriorAlgebra R M →ₐ[R] R :=
                                                       /-
                                                         R : Type u1
                                                         inst✝⁴ : CommRing R
                                                         M : Type u2
                                                         inst✝³ : AddCommGroup M
                                                         inst✝² : Module R M
                                                         A : Type u_1
                                                         inst✝¹ : Semiring A
                                                         inst✝ : Algebra R A
                                                         x✝ : M
                                                         ⊢ Eq (HMul.hMul (0 x✝) (0 x✝)) 0
                                                       -/
  ExteriorAlgebra.lift R ⟨(0 : M →ₗ[R] R), fun _ => by simp⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem algebraMap_leftInverse :
    Function.LeftInverse algebraMapInv (algebraMap R <| ExteriorAlgebra R M) := fun x => by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : R
    ⊢ Eq (ExteriorAlgebra.algebraMapInv ((algebraMap R (ExteriorAlgebra R M)) x)) x
  -/
  simp [algebraMapInv]
  /-
    🎉 no goals
  -/


@[simp]
theorem algebraMap_inj (x y : R) :
    algebraMap R (ExteriorAlgebra R M) x = algebraMap R (ExteriorAlgebra R M) y ↔ x = y :=
  (algebraMap_leftInverse M).injective.eq_iff


@[simp]
theorem algebraMap_eq_zero_iff (x : R) : algebraMap R (ExteriorAlgebra R M) x = 0 ↔ x = 0 :=
  map_eq_zero_iff (algebraMap _ _) (algebraMap_leftInverse _).injective


@[simp]
theorem algebraMap_eq_one_iff (x : R) : algebraMap R (ExteriorAlgebra R M) x = 1 ↔ x = 1 :=
  map_eq_one_iff (algebraMap _ _) (algebraMap_leftInverse _).injective


@[instance]
theorem isLocalHom_algebraMap : IsLocalHom (algebraMap R (ExteriorAlgebra R M)) :=
  isLocalHom_of_leftInverse _ (algebraMap_leftInverse M)


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_algebraMap := isLocalHom_algebraMap


theorem isUnit_algebraMap (r : R) : IsUnit (algebraMap R (ExteriorAlgebra R M) r) ↔ IsUnit r :=
  isUnit_map_of_leftInverse _ (algebraMap_leftInverse M)


/-- Invertibility in the exterior algebra is the same as invertibility of the base ring. -/
@[simps!]
def invertibleAlgebraMapEquiv (r : R) :
    Invertible (algebraMap R (ExteriorAlgebra R M) r) ≃ Invertible r :=
  invertibleEquivOfLeftInverse _ _ _ (algebraMap_leftInverse M)


/-- The canonical map from `ExteriorAlgebra R M` into `TrivSqZeroExt R M` that sends
`ExteriorAlgebra.ι` to `TrivSqZeroExt.inr`. -/
def toTrivSqZeroExt [Module Rᵐᵒᵖ M] [IsCentralScalar R M] :
    ExteriorAlgebra R M →ₐ[R] TrivSqZeroExt R M :=
  lift R ⟨TrivSqZeroExt.inrHom R M, fun m => TrivSqZeroExt.inr_mul_inr R m m⟩


@[simp]
theorem toTrivSqZeroExt_ι [Module Rᵐᵒᵖ M] [IsCentralScalar R M] (x : M) :
    toTrivSqZeroExt (ι R x) = TrivSqZeroExt.inr x :=
  lift_ι_apply _ _ _ _


/-- The left-inverse of `ι`.

As an implementation detail, we implement this using `TrivSqZeroExt` which has a suitable
algebra structure. -/
def ιInv : ExteriorAlgebra R M →ₗ[R] M := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ LinearMap (RingHom.id R) (ExteriorAlgebra R M) M
  -/
  letI : Module Rᵐᵒᵖ M := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    this : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOpposi …
    ⊢ LinearMap (RingHom.id R) (ExteriorAlgebra R M) M
  -/
  haveI : IsCentralScalar R M := ⟨fun r m => rfl⟩
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
    this : IsCentralScalar R M
    ⊢ LinearMap (RingHom.id R) (ExteriorAlgebra R M) M
  -/
  exact (TrivSqZeroExt.sndHom R M).comp toTrivSqZeroExt.toLinearMap
  /-
    🎉 no goals
  -/


theorem ι_leftInverse : Function.LeftInverse ιInv (ι R : M → ExteriorAlgebra R M) := fun x => by
  -- Porting note: Original proof didn't have `letI` and `haveI`
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    ⊢ Eq (ExteriorAlgebra.ιInv ((ExteriorAlgebra.ι R) x)) x
  -/
  letI : Module Rᵐᵒᵖ M := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    this : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOpposi …
    ⊢ Eq (ExteriorAlgebra.ιInv ((ExteriorAlgebra.ι R) x)) x
  -/
  haveI : IsCentralScalar R M := ⟨fun r m => rfl⟩
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
    this : IsCentralScalar R M
    ⊢ Eq (ExteriorAlgebra.ιInv ((ExteriorAlgebra.ι R) x)) x
  -/
  simp [ιInv]
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_inj (x y : M) : ι R x = ι R y ↔ x = y :=
  ι_leftInverse.injective.eq_iff


@[simp]
                                                        /-
                                                          R : Type u1
                                                          inst✝² : CommRing R
                                                          M : Type u2
                                                          inst✝¹ : AddCommGroup M
                                                          inst✝ : Module R M
                                                          x : M
                                                          ⊢ Iff (Eq ((ExteriorAlgebra.ι R) x) 0) (Eq x 0)
                                                        -/
theorem ι_eq_zero_iff (x : M) : ι R x = 0 ↔ x = 0 := by rw [← ι_inj R x 0, LinearMap.map_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem ι_eq_algebraMap_iff (x : M) (r : R) : ι R x = algebraMap R _ r ↔ x = 0 ∧ r = 0 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    r : R
    ⊢ Iff (Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r))  …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
      ⊢ And (Eq x 0) (Eq r 0)
    -/
  · letI : Module Rᵐᵒᵖ M := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
    /-
      case refine_1
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
      this : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOpposi …
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    haveI : IsCentralScalar R M := ⟨fun r m => rfl⟩
    /-
      case refine_1
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
      this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
      this : IsCentralScalar R M
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    have hf0 : toTrivSqZeroExt (ι R x) = (0, x) := toTrivSqZeroExt_ι _
    /-
      case refine_1
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
      this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
      this : IsCentralScalar R M
      hf0 : Eq (ExteriorAlgebra.toTrivSqZeroExt ((ExteriorAlgebra.ι R) x)) { fst :=  …
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    rw [h, AlgHom.commutes] at hf0
    /-
      case refine_1
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
      this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
      this : IsCentralScalar R M
      hf0 : Eq ((algebraMap R (TrivSqZeroExt R M)) r) { fst := 0, snd := x }
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    have : r = 0 ∧ 0 = x := Prod.ext_iff.1 hf0
    /-
      case refine_1
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      h : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
      this✝¹ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppo …
      this✝ : IsCentralScalar R M
      hf0 : Eq ((algebraMap R (TrivSqZeroExt R M)) r) { fst := 0, snd := x }
      this : And (Eq r 0) (Eq 0 x)
      ⊢ And (Eq x 0) (Eq r 0)
    -/
    exact this.symm.imp_left Eq.symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : M
      r : R
      ⊢ And (Eq x 0) (Eq r 0) → Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (Exterio …
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case refine_2.intro
      R : Type u1
      inst✝² : CommRing R
      M : Type u2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      ⊢ Eq ((ExteriorAlgebra.ι R) 0) ((algebraMap R (ExteriorAlgebra R M)) 0)
    -/
    rw [LinearMap.map_zero, RingHom.map_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem ι_ne_one [Nontrivial R] (x : M) : ι R x ≠ 1 := by
  /-
    R : Type u1
    inst✝³ : CommRing R
    M : Type u2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    x : M
    ⊢ Ne ((ExteriorAlgebra.ι R) x) 1
  -/
  rw [← (algebraMap R (ExteriorAlgebra R M)).map_one, Ne, ι_eq_algebraMap_iff]
  /-
    R : Type u1
    inst✝³ : CommRing R
    M : Type u2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    x : M
    ⊢ Not (And (Eq x 0) (Eq 1 0))
  -/
  exact one_ne_zero ∘ And.right
  /-
    🎉 no goals
  -/


/-- The generators of the exterior algebra are disjoint from its scalars. -/
theorem ι_range_disjoint_one :
    Disjoint (LinearMap.range (ι R : M →ₗ[R] ExteriorAlgebra R M))
      (1 : Submodule R (ExteriorAlgebra R M)) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Disjoint (LinearMap.range (ExteriorAlgebra.ι R)) 1
  -/
  rw [Submodule.disjoint_def]
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ ∀ (x : ExteriorAlgebra R M), Membership.mem (LinearMap.range (ExteriorAlgebr …
  -/
  rintro _ ⟨x, hx⟩ h
  /-
    case intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝ : ExteriorAlgebra R M
    x : M
    hx : Eq ((ExteriorAlgebra.ι R) x) x✝
    h : Membership.mem 1 x✝
    ⊢ Eq x✝ 0
  -/
  obtain ⟨r, rfl : algebraMap R (ExteriorAlgebra R M) r = _⟩ := Submodule.mem_one.mp h
  /-
    case intro.intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    r : R
    hx : Eq ((ExteriorAlgebra.ι R) x) ((algebraMap R (ExteriorAlgebra R M)) r)
    h : Membership.mem 1 ((algebraMap R (ExteriorAlgebra R M)) r)
    ⊢ Eq ((algebraMap R (ExteriorAlgebra R M)) r) 0
  -/
  rw [ι_eq_algebraMap_iff x] at hx
  /-
    case intro.intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    r : R
    hx : And (Eq x 0) (Eq r 0)
    h : Membership.mem 1 ((algebraMap R (ExteriorAlgebra R M)) r)
    ⊢ Eq ((algebraMap R (ExteriorAlgebra R M)) r) 0
  -/
  rw [hx.2, RingHom.map_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_add_mul_swap (x y : M) : ι R x * ι R y + ι R y * ι R x = 0 :=
  CliffordAlgebra.ι_mul_ι_add_swap_of_isOrtho <| .all _ _


theorem ι_mul_prod_list {n : ℕ} (f : Fin n → M) (i : Fin n) :
    (ι R <| f i) * (List.ofFn fun i => ι R <| f i).prod = 0 := by
  induction n with
  | zero => exact i.elim0
  | succ n hn =>
    rw [List.ofFn_succ, List.prod_cons, ← mul_assoc]
    by_cases h : i = 0
    · rw [h, ι_sq_zero, zero_mul]
    · replace hn :=
        congr_arg (ι R (f 0) * ·) <| hn (fun i => f <| Fin.succ i) (i.pred h)
      simp only at hn
      rw [Fin.succ_pred, ← mul_assoc, mul_zero] at hn
      refine (eq_zero_iff_eq_zero_of_add_eq_zero ?_).mp hn
      rw [← add_mul, ι_add_mul_swap, zero_mul]


/-- The product of `n` terms of the form `ι R m` is an alternating map.

This is a special case of `MultilinearMap.mkPiAlgebraFin`, and the exterior algebra version of
`TensorAlgebra.tprod`. -/
def ιMulti (n : ℕ) : M [⋀^Fin n]→ₗ[R] ExteriorAlgebra R M :=
  let F := (MultilinearMap.mkPiAlgebraFin R n (ExteriorAlgebra R M)).compLinearMap fun _ => ι R
  { F with
    map_eq_zero_of_eq' := fun f x y hfxy hxy => by
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        F : MultilinearMap R (fun x => M) (ExteriorAlgebra R M) := (MultilinearMap.mkP …
        f : Fin n → M
        x y : Fin n
        hfxy : Eq (f x) (f y)
        hxy : Ne x y
        ⊢ Eq ({ toFun := ⇑F, map_update_add' := ⋯, map_update_smul' := ⋯ }.toFun f) 0
      -/
      dsimp [F]
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        F : MultilinearMap R (fun x => M) (ExteriorAlgebra R M) := (MultilinearMap.mkP …
        f : Fin n → M
        x y : Fin n
        hfxy : Eq (f x) (f y)
        hxy : Ne x y
        ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
      -/
      clear F
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        f : Fin n → M
        x y : Fin n
        hfxy : Eq (f x) (f y)
        hxy : Ne x y
        ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
      -/
      wlog h : x < y
        /-
          case inr
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          n : Nat
          f : Fin n → M
          x y : Fin n
          hfxy : Eq (f x) (f y)
          hxy : Ne x y
          this : ∀ (R : Type u1) [inst : CommRing R] {M : Type u2} [inst_1 : AddCommGrou …
          h : Not (LT.lt x y)
          ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
        -/
      · exact this R n f y x hfxy.symm hxy.symm (hxy.lt_or_lt.resolve_left h)
        /-
          🎉 no goals
        -/
      /-
        R✝ : Type u1
        inst✝⁵ : CommRing R✝
        M✝ : Type u2
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        f : Fin n → M
        x y : Fin n
        hfxy : Eq (f x) (f y)
        hxy : Ne x y
        h : LT.lt x y
        ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
      -/
      clear hxy
      /-
        R✝ : Type u1
        inst✝⁵ : CommRing R✝
        M✝ : Type u2
        inst✝⁴ : AddCommGroup M✝
        inst✝³ : Module R✝ M✝
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        f : Fin n → M
        x y : Fin n
        hfxy : Eq (f x) (f y)
        h : LT.lt x y
        ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
      -/
      induction' n with n hn
        /-
          case zero
          R✝ : Type u1
          inst✝⁵ : CommRing R✝
          M✝ : Type u2
          inst✝⁴ : AddCommGroup M✝
          inst✝³ : Module R✝ M✝
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          f : Fin 0 → M
          x y : Fin 0
          hfxy : Eq (f x) (f y)
          h : LT.lt x y
          ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
        -/
      · exact x.elim0
        /-
          🎉 no goals
        -/
        /-
          case succ
          R✝ : Type u1
          inst✝⁵ : CommRing R✝
          M✝ : Type u2
          inst✝⁴ : AddCommGroup M✝
          inst✝³ : Module R✝ M✝
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          n : Nat
          hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
          f : Fin (HAdd.hAdd n 1) → M
          x y : Fin (HAdd.hAdd n 1)
          hfxy : Eq (f x) (f y)
          h : LT.lt x y
          ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (f i)).prod 0
        -/
      · rw [List.ofFn_succ, List.prod_cons]
        /-
          case succ
          R✝ : Type u1
          inst✝⁵ : CommRing R✝
          M✝ : Type u2
          inst✝⁴ : AddCommGroup M✝
          inst✝³ : Module R✝ M✝
          R : Type u1
          inst✝² : CommRing R
          M : Type u2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          n : Nat
          hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
          f : Fin (HAdd.hAdd n 1) → M
          x y : Fin (HAdd.hAdd n 1)
          hfxy : Eq (f x) (f y)
          h : LT.lt x y
          ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (f 0)) (List.ofFn fun i => (ExteriorAlg …
        -/
        by_cases hx : x = 0
        -- one of the repeated terms is on the left
          /-
            case pos
            R✝ : Type u1
            inst✝⁵ : CommRing R✝
            M✝ : Type u2
            inst✝⁴ : AddCommGroup M✝
            inst✝³ : Module R✝ M✝
            R : Type u1
            inst✝² : CommRing R
            M : Type u2
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            n : Nat
            hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
            f : Fin (HAdd.hAdd n 1) → M
            x y : Fin (HAdd.hAdd n 1)
            hfxy : Eq (f x) (f y)
            h : LT.lt x y
            hx : Eq x 0
            ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (f 0)) (List.ofFn fun i => (ExteriorAlg …
          -/
        · rw [hx] at hfxy h
          /-
            case pos
            R✝ : Type u1
            inst✝⁵ : CommRing R✝
            M✝ : Type u2
            inst✝⁴ : AddCommGroup M✝
            inst✝³ : Module R✝ M✝
            R : Type u1
            inst✝² : CommRing R
            M : Type u2
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            n : Nat
            hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
            f : Fin (HAdd.hAdd n 1) → M
            x y : Fin (HAdd.hAdd n 1)
            hfxy : Eq (f 0) (f y)
            h : LT.lt 0 y
            hx : Eq x 0
            ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (f 0)) (List.ofFn fun i => (ExteriorAlg …
          -/
          rw [hfxy, ← Fin.succ_pred y (ne_of_lt h).symm]
          /-
            case pos
            R✝ : Type u1
            inst✝⁵ : CommRing R✝
            M✝ : Type u2
            inst✝⁴ : AddCommGroup M✝
            inst✝³ : Module R✝ M✝
            R : Type u1
            inst✝² : CommRing R
            M : Type u2
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            n : Nat
            hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
            f : Fin (HAdd.hAdd n 1) → M
            x y : Fin (HAdd.hAdd n 1)
            hfxy : Eq (f 0) (f y)
            h : LT.lt 0 y
            hx : Eq x 0
            ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (f (y.pred ⋯).succ)) (List.ofFn fun i = …
          -/
          exact ι_mul_prod_list (f ∘ Fin.succ) _
          /-
            🎉 no goals
          -/
        -- ignore the left-most term and induct on the remaining ones, decrementing indices
          /-
            case neg
            R✝ : Type u1
            inst✝⁵ : CommRing R✝
            M✝ : Type u2
            inst✝⁴ : AddCommGroup M✝
            inst✝³ : Module R✝ M✝
            R : Type u1
            inst✝² : CommRing R
            M : Type u2
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            n : Nat
            hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
            f : Fin (HAdd.hAdd n 1) → M
            x y : Fin (HAdd.hAdd n 1)
            hfxy : Eq (f x) (f y)
            h : LT.lt x y
            hx : Not (Eq x 0)
            ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (f 0)) (List.ofFn fun i => (ExteriorAlg …
          -/
        · convert mul_zero (ι R (f 0))
          refine
            hn
              (fun i => f <| Fin.succ i) (x.pred hx)
              (y.pred (ne_of_lt <| lt_of_le_of_lt x.zero_le h).symm) ?_
              (Fin.pred_lt_pred_iff.mpr h)
          /-
            case h.e'_2.h.e'_6
            R✝ : Type u1
            inst✝⁵ : CommRing R✝
            M✝ : Type u2
            inst✝⁴ : AddCommGroup M✝
            inst✝³ : Module R✝ M✝
            R : Type u1
            inst✝² : CommRing R
            M : Type u2
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            n : Nat
            hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
            f : Fin (HAdd.hAdd n 1) → M
            x y : Fin (HAdd.hAdd n 1)
            hfxy : Eq (f x) (f y)
            h : LT.lt x y
            hx : Not (Eq x 0)
            ⊢ Eq ((fun i => f i.succ) (x.pred hx)) ((fun i => f i.succ) (y.pred ⋯))
          -/
          simp only [Fin.succ_pred]
          /-
            case h.e'_2.h.e'_6
            R✝ : Type u1
            inst✝⁵ : CommRing R✝
            M✝ : Type u2
            inst✝⁴ : AddCommGroup M✝
            inst✝³ : Module R✝ M✝
            R : Type u1
            inst✝² : CommRing R
            M : Type u2
            inst✝¹ : AddCommGroup M
            inst✝ : Module R M
            n : Nat
            hn : ∀ (f : Fin n → M) (x y : Fin n), Eq (f x) (f y) → LT.lt x y → Eq (List.of …
            f : Fin (HAdd.hAdd n 1) → M
            x y : Fin (HAdd.hAdd n 1)
            hfxy : Eq (f x) (f y)
            h : LT.lt x y
            hx : Not (Eq x 0)
            ⊢ Eq (f x) (f y)
          -/
          exact hfxy
          /-
            🎉 no goals
          -/
    toFun := F }


theorem ιMulti_apply {n : ℕ} (v : Fin n → M) : ιMulti R n v = (List.ofFn fun i => ι R (v i)).prod :=
  rfl


@[simp]
theorem ιMulti_zero_apply (v : Fin 0 → M) : ιMulti R 0 v = 1 := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    v : Fin 0 → M
    ⊢ Eq ((ExteriorAlgebra.ιMulti R 0) v) 1
  -/
  simp [ιMulti]
  /-
    🎉 no goals
  -/


@[simp]
theorem ιMulti_succ_apply {n : ℕ} (v : Fin n.succ → M) :
    ιMulti R _ v = ι R (v 0) * ιMulti R _ (Matrix.vecTail v) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    v : Fin n.succ → M
    ⊢ Eq ((ExteriorAlgebra.ιMulti R n.succ) v) (HMul.hMul ((ExteriorAlgebra.ι R) ( …
  -/
  simp [ιMulti, Matrix.vecTail]
  /-
    🎉 no goals
  -/


theorem ιMulti_succ_curryLeft {n : ℕ} (m : M) :
    (ιMulti R n.succ).curryLeft m = (LinearMap.mulLeft R (ι R m)).compAlternatingMap (ιMulti R n) :=
  AlternatingMap.ext fun v =>
    (ιMulti_succ_apply _).trans <| by
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        m : M
        v : Fin n → M
        ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (Matrix.vecCons m v 0)) ((ExteriorAlgeb …
      -/
      simp_rw [Matrix.tail_cons]
      /-
        R : Type u1
        inst✝² : CommRing R
        M : Type u2
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        n : Nat
        m : M
        v : Fin n → M
        ⊢ Eq (HMul.hMul ((ExteriorAlgebra.ι R) (Matrix.vecCons m v 0)) ((ExteriorAlgeb …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- The image of `ExteriorAlgebra.ιMulti R n` is contained in the `n`th exterior power. -/
lemma ιMulti_range (n : ℕ) :
    Set.range (ιMulti R n (M := M)) ⊆ ↑(⋀[R]^n M) := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ HasSubset.Subset (Set.range ⇑(ExteriorAlgebra.ιMulti R n)) ↑(ExteriorAlgebra …
  -/
  rw [Set.range_subset_iff]
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ ∀ (y : Fin n → M), Membership.mem (↑(ExteriorAlgebra.exteriorPower R n M)) ( …
  -/
  intro v
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    v : Fin n → M
    ⊢ Membership.mem (↑(ExteriorAlgebra.exteriorPower R n M)) ((ExteriorAlgebra.ιM …
  -/
  rw [ιMulti_apply]
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    v : Fin n → M
    ⊢ Membership.mem (↑(ExteriorAlgebra.exteriorPower R n M)) (List.ofFn fun i =>  …
  -/
  apply Submodule.pow_subset_pow
  /-
    case a
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    v : Fin n → M
    ⊢ Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) (Lis …
  -/
  rw [Set.mem_pow]
  /-
    case a
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    v : Fin n → M
    ⊢ Exists fun f => Eq (List.ofFn fun i => ↑(f i)).prod (List.ofFn fun i => (Ext …
  -/
  exact ⟨fun i => ⟨ι R (v i), LinearMap.mem_range_self _ _⟩, rfl⟩
  /-
    🎉 no goals
  -/


/-- The image of `ExteriorAlgebra.ιMulti R n` spans the `n`th exterior power, as a submodule
of the exterior algebra. -/
lemma ιMulti_span_fixedDegree (n : ℕ) :
    Submodule.span R (Set.range (ιMulti R n)) = ⋀[R]^n M := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ Eq (Submodule.span R (Set.range ⇑(ExteriorAlgebra.ιMulti R n))) (ExteriorAlg …
  -/
  refine le_antisymm (Submodule.span_le.2 (ιMulti_range R n)) ?_
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ LE.le (ExteriorAlgebra.exteriorPower R n M) (Submodule.span R (Set.range ⇑(E …
  -/
  rw [exteriorPower, Submodule.pow_eq_span_pow_set, Submodule.span_le]
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ HasSubset.Subset (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) ↑( …
  -/
  refine fun u hu ↦ Submodule.subset_span ?_
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    u : ExteriorAlgebra R M
    hu : Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) u
    ⊢ Membership.mem (Set.range ⇑(ExteriorAlgebra.ιMulti R n)) u
  -/
  obtain ⟨f, rfl⟩ := Set.mem_pow.mp hu
  /-
    case intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    f : Fin n → ↑↑(LinearMap.range (ExteriorAlgebra.ι R))
    hu : Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) ( …
    ⊢ Membership.mem (Set.range ⇑(ExteriorAlgebra.ιMulti R n)) (List.ofFn fun i => …
  -/
  refine ⟨fun i => ιInv (f i).1, ?_⟩
  /-
    case intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    f : Fin n → ↑↑(LinearMap.range (ExteriorAlgebra.ι R))
    hu : Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) ( …
    ⊢ Eq ((ExteriorAlgebra.ιMulti R n) fun i => ExteriorAlgebra.ιInv ↑(f i)) (List …
  -/
  rw [ιMulti_apply]
  /-
    case intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    f : Fin n → ↑↑(LinearMap.range (ExteriorAlgebra.ι R))
    hu : Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) ( …
    ⊢ Eq (List.ofFn fun i => (ExteriorAlgebra.ι R) (ExteriorAlgebra.ιInv ↑(f i))). …
  -/
  congr with i
  /-
    case intro.e_a.e_f.h
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    f : Fin n → ↑↑(LinearMap.range (ExteriorAlgebra.ι R))
    hu : Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) ( …
    i : Fin n
    ⊢ Eq ((ExteriorAlgebra.ι R) (ExteriorAlgebra.ιInv ↑(f i))) ↑(f i)
  -/
  obtain ⟨v, hv⟩ := (f i).prop
  /-
    case intro.e_a.e_f.h.intro
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    f : Fin n → ↑↑(LinearMap.range (ExteriorAlgebra.ι R))
    hu : Membership.mem (HPow.hPow (↑(LinearMap.range (ExteriorAlgebra.ι R))) n) ( …
    i : Fin n
    v : M
    hv : Eq ((ExteriorAlgebra.ι R) v) ↑(f i)
    ⊢ Eq ((ExteriorAlgebra.ι R) (ExteriorAlgebra.ιInv ↑(f i))) ↑(f i)
  -/
  rw [← hv, ι_leftInverse]
  /-
    🎉 no goals
  -/


/-- Given a linearly ordered family `v` of vectors of `M` and a natural number `n`, produce the
family of `n`fold exterior products of elements of `v`, seen as members of the exterior algebra. -/
abbrev ιMulti_family (n : ℕ) {I : Type*} [LinearOrder I] (v : I → M)
    (s : {s : Finset I // Finset.card s = n}) : ExteriorAlgebra R M :=
  ιMulti R n fun i => v (Finset.orderIsoOfFin _ s.prop i)


/-- An `ExteriorAlgebra` over a nontrivial ring is nontrivial. -/
instance [Nontrivial R] : Nontrivial (ExteriorAlgebra R M) :=
  (algebraMap_leftInverse M).injective.nontrivial


/-- The morphism of exterior algebras induced by a linear map. -/
def map (f : M →ₗ[R] N) : ExteriorAlgebra R M →ₐ[R] ExteriorAlgebra R N :=
  CliffordAlgebra.map { f with map_app' := fun _ => rfl }


@[simp]
theorem map_comp_ι (f : M →ₗ[R] N) : (map f).toLinearMap ∘ₗ ι R = ι R ∘ₗ f :=
  CliffordAlgebra.map_comp_ι _


@[simp]
theorem map_apply_ι (f : M →ₗ[R] N) (m : M) : map f (ι R m) = ι R (f m) :=
  CliffordAlgebra.map_apply_ι _ m


@[simp]
theorem map_apply_ιMulti {n : ℕ} (f : M →ₗ[R] N) (m : Fin n → M) :
    map f (ιMulti R n m) = ιMulti R n (f ∘ m) := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    n : Nat
    f : LinearMap (RingHom.id R) M N
    m : Fin n → M
    ⊢ Eq ((ExteriorAlgebra.map f) ((ExteriorAlgebra.ιMulti R n) m)) ((ExteriorAlge …
  -/
  rw [ιMulti_apply, ιMulti_apply, map_list_prod]
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    n : Nat
    f : LinearMap (RingHom.id R) M N
    m : Fin n → M
    ⊢ Eq (List.map (⇑(ExteriorAlgebra.map f)) (List.ofFn fun i => (ExteriorAlgebra …
  -/
  simp only [List.map_ofFn, Function.comp_def, map_apply_ι]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_comp_ιMulti {n : ℕ} (f : M →ₗ[R] N) :
    (map f).toLinearMap.compAlternatingMap (ιMulti R n (M := M)) =
    (ιMulti R n (M := N)).compLinearMap f := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    n : Nat
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq ((ExteriorAlgebra.map f).toLinearMap.compAlternatingMap (ExteriorAlgebra. …
  -/
  ext m
  /-
    case H
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    n : Nat
    f : LinearMap (RingHom.id R) M N
    m : Fin n → M
    ⊢ Eq (((ExteriorAlgebra.map f).toLinearMap.compAlternatingMap (ExteriorAlgebra …
  -/
  exact map_apply_ιMulti _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem map_id :
    map LinearMap.id = AlgHom.id R (ExteriorAlgebra R M) :=
  CliffordAlgebra.map_id 0


@[simp]
theorem map_comp_map (f : M →ₗ[R] N) (g : N →ₗ[R] N') :
    AlgHom.comp (map g) (map f) = map (LinearMap.comp g f) :=
  CliffordAlgebra.map_comp_map _ _


@[simp]
theorem ι_range_map_map (f : M →ₗ[R] N) :
    Submodule.map (AlgHom.toLinearMap (map f)) (LinearMap.range (ι R (M := M))) =
    Submodule.map (ι R) (LinearMap.range f) :=
  CliffordAlgebra.ι_range_map_map _


theorem toTrivSqZeroExt_comp_map [Module Rᵐᵒᵖ M] [IsCentralScalar R M] [Module Rᵐᵒᵖ N]
    [IsCentralScalar R N] (f : M →ₗ[R] N) :
    toTrivSqZeroExt.comp (map f) = (TrivSqZeroExt.map f).comp toTrivSqZeroExt := by
  /-
    R : Type u1
    inst✝⁸ : CommRing R
    M : Type u2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u4
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module (MulOpposite R) M
    inst✝² : IsCentralScalar R M
    inst✝¹ : Module (MulOpposite R) N
    inst✝ : IsCentralScalar R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq (ExteriorAlgebra.toTrivSqZeroExt.comp (ExteriorAlgebra.map f)) ((TrivSqZe …
  -/
  apply hom_ext
  /-
    case h
    R : Type u1
    inst✝⁸ : CommRing R
    M : Type u2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u4
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module (MulOpposite R) M
    inst✝² : IsCentralScalar R M
    inst✝¹ : Module (MulOpposite R) N
    inst✝ : IsCentralScalar R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq ((ExteriorAlgebra.toTrivSqZeroExt.comp (ExteriorAlgebra.map f)).toLinearM …
  -/
  apply LinearMap.ext
  simp only [AlgHom.comp_toLinearMap, LinearMap.coe_comp, Function.comp_apply,
    AlgHom.toLinearMap_apply, map_apply_ι, toTrivSqZeroExt_ι, TrivSqZeroExt.map_inr, forall_const]


theorem ιInv_comp_map (f : M →ₗ[R] N) :
    ιInv.comp (map f).toLinearMap = f.comp ιInv := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq (ExteriorAlgebra.ιInv.comp (ExteriorAlgebra.map f).toLinearMap) (f.comp E …
  -/
  letI : Module Rᵐᵒᵖ M := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    this : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOpposi …
    ⊢ Eq (ExteriorAlgebra.ιInv.comp (ExteriorAlgebra.map f).toLinearMap) (f.comp E …
  -/
  haveI : IsCentralScalar R M := ⟨fun r m => rfl⟩
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    this✝ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppos …
    this : IsCentralScalar R M
    ⊢ Eq (ExteriorAlgebra.ιInv.comp (ExteriorAlgebra.map f).toLinearMap) (f.comp E …
  -/
  letI : Module Rᵐᵒᵖ N := Module.compHom _ ((RingHom.id R).fromOpposite mul_comm)
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    this✝¹ : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppo …
    this✝ : IsCentralScalar R M
    this : Module (MulOpposite R) N := Module.compHom N ((RingHom.id R).fromOpposi …
    ⊢ Eq (ExteriorAlgebra.ιInv.comp (ExteriorAlgebra.map f).toLinearMap) (f.comp E …
  -/
  haveI : IsCentralScalar R N := ⟨fun r m => rfl⟩
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    this✝² : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppo …
    this✝¹ : IsCentralScalar R M
    this✝ : Module (MulOpposite R) N := Module.compHom N ((RingHom.id R).fromOppos …
    this : IsCentralScalar R N
    ⊢ Eq (ExteriorAlgebra.ιInv.comp (ExteriorAlgebra.map f).toLinearMap) (f.comp E …
  -/
  unfold ιInv
  conv_lhs => rw [LinearMap.comp_assoc, ← AlgHom.comp_toLinearMap, toTrivSqZeroExt_comp_map,
                AlgHom.comp_toLinearMap, ← LinearMap.comp_assoc, TrivSqZeroExt.sndHom_comp_map]
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    this✝² : Module (MulOpposite R) M := Module.compHom M ((RingHom.id R).fromOppo …
    this✝¹ : IsCentralScalar R M
    this✝ : Module (MulOpposite R) N := Module.compHom N ((RingHom.id R).fromOppos …
    this : IsCentralScalar R N
    ⊢ Eq ((f.comp (TrivSqZeroExt.sndHom R M)).comp ExteriorAlgebra.toTrivSqZeroExt …
  -/
  rfl
  /-
    🎉 no goals
  -/


open Function in
/-- For a linear map `f` from `M` to `N`,
`ExteriorAlgebra.map g` is a retraction of `ExteriorAlgebra.map f` iff
`g` is a retraction of `f`. -/
@[simp]
lemma leftInverse_map_iff {f : M →ₗ[R] N} {g : N →ₗ[R] M} :
    LeftInverse (map g) (map f) ↔ LeftInverse g f := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N M
    ⊢ Iff (Function.LeftInverse ⇑(ExteriorAlgebra.map g) ⇑(ExteriorAlgebra.map f)) …
  -/
  refine ⟨fun h x => ?_, fun h => CliffordAlgebra.leftInverse_map_of_leftInverse _ _ h⟩
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N M
    h : Function.LeftInverse ⇑(ExteriorAlgebra.map g) ⇑(ExteriorAlgebra.map f)
    x : M
    ⊢ Eq (g (f x)) x
  -/
  simpa using h (ι _ x)
  /-
    🎉 no goals
  -/


/-- A morphism of modules that admits a linear retraction induces an injective morphism of
exterior algebras. -/
lemma map_injective {f : M →ₗ[R] N} (hf : ∃ (g : N →ₗ[R] M), g.comp f = LinearMap.id) :
    Function.Injective (map f) :=
  let ⟨_, hgf⟩ := hf; (leftInverse_map_iff.mpr (DFunLike.congr_fun hgf)).injective


/-- A morphism of modules is surjective if and only the morphism of exterior algebras that it
induces is surjective. -/
@[simp]
lemma map_surjective_iff {f : M →ₗ[R] N} :
    Function.Surjective (map f) ↔ Function.Surjective f := by
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Iff (Function.Surjective ⇑(ExteriorAlgebra.map f)) (Function.Surjective ⇑f)
  -/
  refine ⟨fun h y ↦ ?_, fun h ↦ CliffordAlgebra.map_surjective _ h⟩
  /-
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    h : Function.Surjective ⇑(ExteriorAlgebra.map f)
    y : N
    ⊢ Exists fun a => Eq (f a) y
  -/
  obtain ⟨x, hx⟩ := h (ι R y)
  /-
    case intro
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    h : Function.Surjective ⇑(ExteriorAlgebra.map f)
    y : N
    x : ExteriorAlgebra R M
    hx : Eq ((ExteriorAlgebra.map f) x) ((ExteriorAlgebra.ι R) y)
    ⊢ Exists fun a => Eq (f a) y
  -/
  existsi ιInv x
  /-
    case intro
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    h : Function.Surjective ⇑(ExteriorAlgebra.map f)
    y : N
    x : ExteriorAlgebra R M
    hx : Eq ((ExteriorAlgebra.map f) x) ((ExteriorAlgebra.ι R) y)
    ⊢ Eq (f (ExteriorAlgebra.ιInv x)) y
  -/
  rw [← LinearMap.comp_apply, ← ιInv_comp_map, LinearMap.comp_apply]
  /-
    case intro
    R : Type u1
    inst✝⁴ : CommRing R
    M : Type u2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u4
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    h : Function.Surjective ⇑(ExteriorAlgebra.map f)
    y : N
    x : ExteriorAlgebra R M
    hx : Eq ((ExteriorAlgebra.map f) x) ((ExteriorAlgebra.ι R) y)
    ⊢ Eq (ExteriorAlgebra.ιInv ((ExteriorAlgebra.map f).toLinearMap x)) y
  -/
  erw [hx, ExteriorAlgebra.ι_leftInverse]
  /-
    🎉 no goals
  -/


/-- An injective morphism of vector spaces induces an injective morphism of exterior algebras. -/
lemma map_injective_field {f : E →ₗ[K] F} (hf : LinearMap.ker f = ⊥) :
    Function.Injective (map f) :=
  map_injective (LinearMap.exists_leftInverse_of_injective f hf)


/-- The canonical image of the `TensorAlgebra` in the `ExteriorAlgebra`, which maps
`TensorAlgebra.ι R x` to `ExteriorAlgebra.ι R x`. -/
def toExterior : TensorAlgebra R M →ₐ[R] ExteriorAlgebra R M :=
  TensorAlgebra.lift R (ExteriorAlgebra.ι R : M →ₗ[R] ExteriorAlgebra R M)


@[simp]
theorem toExterior_ι (m : M) :
    TensorAlgebra.toExterior (TensorAlgebra.ι R m) = ExteriorAlgebra.ι R m := by
  /-
    R : Type u1
    inst✝² : CommRing R
    M : Type u2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    m : M
    ⊢ Eq (TensorAlgebra.toExterior ((TensorAlgebra.ι R) m)) ((ExteriorAlgebra.ι R) …
  -/
  simp [toExterior]
  /-
    🎉 no goals
  -/


