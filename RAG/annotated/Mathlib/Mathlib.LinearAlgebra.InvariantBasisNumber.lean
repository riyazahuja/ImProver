/-- We say that `R` satisfies the strong rank condition if `(Fin n → R) →ₗ[R] (Fin m → R)` injective
    implies `n ≤ m`. -/
@[mk_iff]
class StrongRankCondition : Prop where
  /-- Any injective linear map from `Rⁿ` to `Rᵐ` guarantees `n ≤ m`. -/
  le_of_fin_injective : ∀ {n m : ℕ} (f : (Fin n → R) →ₗ[R] Fin m → R), Injective f → n ≤ m


theorem le_of_fin_injective [StrongRankCondition R] {n m : ℕ} (f : (Fin n → R) →ₗ[R] Fin m → R) :
    Injective f → n ≤ m :=
  StrongRankCondition.le_of_fin_injective f


/-- A ring satisfies the strong rank condition if and only if, for all `n : ℕ`, any linear map
`(Fin (n + 1) → R) →ₗ[R] (Fin n → R)` is not injective. -/
theorem strongRankCondition_iff_succ :
    StrongRankCondition R ↔
      ∀ (n : ℕ) (f : (Fin (n + 1) → R) →ₗ[R] Fin n → R), ¬Function.Injective f := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Iff (StrongRankCondition R) (∀ (n : Nat) (f : LinearMap (RingHom.id R) (Fin  …
  -/
  refine ⟨fun h n => fun f hf => ?_, fun h => ⟨@fun n m f hf => ?_⟩⟩
    /-
      case refine_1
      R : Type u
      inst✝ : Semiring R
      h : StrongRankCondition R
      n : Nat
      f : LinearMap (RingHom.id R) (Fin (HAdd.hAdd n 1) → R) (Fin n → R)
      hf : Function.Injective ⇑f
      ⊢ False
    -/
  · letI : StrongRankCondition R := h
    /-
      case refine_1
      R : Type u
      inst✝ : Semiring R
      h : StrongRankCondition R
      n : Nat
      f : LinearMap (RingHom.id R) (Fin (HAdd.hAdd n 1) → R) (Fin n → R)
      hf : Function.Injective ⇑f
      this : StrongRankCondition R := h
      ⊢ False
    -/
    exact Nat.not_succ_le_self n (le_of_fin_injective R f hf)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : Semiring R
      h : ∀ (n : Nat) (f : LinearMap (RingHom.id R) (Fin (HAdd.hAdd n 1) → R) (Fin n …
      n m : Nat
      f : LinearMap (RingHom.id R) (Fin n → R) (Fin m → R)
      hf : Function.Injective ⇑f
      ⊢ LE.le n m
    -/
  · by_contra H
    exact
      h m (f.comp (Function.ExtendByZero.linearMap R (Fin.castLE (not_le.1 H))))
        (hf.comp (Function.extend_injective (Fin.strictMono_castLE _).injective _))


/-- Any nontrivial ring satisfying Orzech property also satisfies strong rank condition. -/
instance (priority := 100) strongRankCondition_of_orzechProperty
    [Nontrivial R] [OrzechProperty R] : StrongRankCondition R := by
  /-
    R : Type u
    inst✝² : Semiring R
    inst✝¹ : Nontrivial R
    inst✝ : OrzechProperty R
    ⊢ StrongRankCondition R
  -/
  refine (strongRankCondition_iff_succ R).2 fun n i hi ↦ ?_
  let f : (Fin (n + 1) → R) →ₗ[R] Fin n → R := {
    toFun := fun x ↦ x ∘ Fin.castSucc
    map_add' := fun _ _ ↦ rfl
    map_smul' := fun _ _ ↦ rfl
  }
  have h : (0 : Fin (n + 1) → R) = update (0 : Fin (n + 1) → R) (Fin.last n) 1 := by
    apply OrzechProperty.injective_of_surjective_of_injective i f hi
      (Fin.castSucc_injective _).surjective_comp_right
    ext m
    simp [f, update_apply, (Fin.castSucc_lt_last m).ne]
  /-
    R : Type u
    inst✝² : Semiring R
    inst✝¹ : Nontrivial R
    inst✝ : OrzechProperty R
    n : Nat
    i : LinearMap (RingHom.id R) (Fin (HAdd.hAdd n 1) → R) (Fin n → R)
    hi : Function.Injective ⇑i
    f : LinearMap (RingHom.id R) (Fin (HAdd.hAdd n 1) → R) (Fin n → R) := { toFun  …
    h : Eq 0 (Function.update 0 (Fin.last n) 1)
    ⊢ False
  -/
  simpa using congr_fun h (Fin.last n)
  /-
    🎉 no goals
  -/


theorem card_le_of_injective [StrongRankCondition R] {α β : Type*} [Fintype α] [Fintype β]
    (f : (α → R) →ₗ[R] β → R) (i : Injective f) : Fintype.card α ≤ Fintype.card β := by
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : StrongRankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (α → R) (β → R)
    i : Function.Injective ⇑f
    ⊢ LE.le (Fintype.card α) (Fintype.card β)
  -/
  let P := LinearEquiv.funCongrLeft R R (Fintype.equivFin α)
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : StrongRankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (α → R) (β → R)
    i : Function.Injective ⇑f
    P : LinearEquiv (RingHom.id R) (Fin (Fintype.card α) → R) (α → R) := LinearEqu …
    ⊢ LE.le (Fintype.card α) (Fintype.card β)
  -/
  let Q := LinearEquiv.funCongrLeft R R (Fintype.equivFin β)
  exact
    le_of_fin_injective R ((Q.symm.toLinearMap.comp f).comp P.toLinearMap)
      (((LinearEquiv.symm Q).injective.comp i).comp (LinearEquiv.injective P))


theorem card_le_of_injective' [StrongRankCondition R] {α β : Type*} [Fintype α] [Fintype β]
    (f : (α →₀ R) →ₗ[R] β →₀ R) (i : Injective f) : Fintype.card α ≤ Fintype.card β := by
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : StrongRankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (Finsupp α R) (Finsupp β R)
    i : Function.Injective ⇑f
    ⊢ LE.le (Fintype.card α) (Fintype.card β)
  -/
  let P := Finsupp.linearEquivFunOnFinite R R β
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : StrongRankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (Finsupp α R) (Finsupp β R)
    i : Function.Injective ⇑f
    P : LinearEquiv (RingHom.id R) (Finsupp β R) (β → R) := Finsupp.linearEquivFun …
    ⊢ LE.le (Fintype.card α) (Fintype.card β)
  -/
  let Q := (Finsupp.linearEquivFunOnFinite R R α).symm
  exact
    card_le_of_injective R ((P.toLinearMap.comp f).comp Q.toLinearMap)
      ((P.injective.comp i).comp Q.injective)


/-- We say that `R` satisfies the rank condition if `(Fin n → R) →ₗ[R] (Fin m → R)` surjective
    implies `m ≤ n`. -/
class RankCondition : Prop where
  /-- Any surjective linear map from `Rⁿ` to `Rᵐ` guarantees `m ≤ n`. -/
  le_of_fin_surjective : ∀ {n m : ℕ} (f : (Fin n → R) →ₗ[R] Fin m → R), Surjective f → m ≤ n


theorem le_of_fin_surjective [RankCondition R] {n m : ℕ} (f : (Fin n → R) →ₗ[R] Fin m → R) :
    Surjective f → m ≤ n :=
  RankCondition.le_of_fin_surjective f


theorem card_le_of_surjective [RankCondition R] {α β : Type*} [Fintype α] [Fintype β]
    (f : (α → R) →ₗ[R] β → R) (i : Surjective f) : Fintype.card β ≤ Fintype.card α := by
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : RankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (α → R) (β → R)
    i : Function.Surjective ⇑f
    ⊢ LE.le (Fintype.card β) (Fintype.card α)
  -/
  let P := LinearEquiv.funCongrLeft R R (Fintype.equivFin α)
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : RankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (α → R) (β → R)
    i : Function.Surjective ⇑f
    P : LinearEquiv (RingHom.id R) (Fin (Fintype.card α) → R) (α → R) := LinearEqu …
    ⊢ LE.le (Fintype.card β) (Fintype.card α)
  -/
  let Q := LinearEquiv.funCongrLeft R R (Fintype.equivFin β)
  exact
    le_of_fin_surjective R ((Q.symm.toLinearMap.comp f).comp P.toLinearMap)
      (((LinearEquiv.symm Q).surjective.comp i).comp (LinearEquiv.surjective P))


theorem card_le_of_surjective' [RankCondition R] {α β : Type*} [Fintype α] [Fintype β]
    (f : (α →₀ R) →ₗ[R] β →₀ R) (i : Surjective f) : Fintype.card β ≤ Fintype.card α := by
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : RankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (Finsupp α R) (Finsupp β R)
    i : Function.Surjective ⇑f
    ⊢ LE.le (Fintype.card β) (Fintype.card α)
  -/
  let P := Finsupp.linearEquivFunOnFinite R R β
  /-
    R : Type u
    inst✝³ : Semiring R
    inst✝² : RankCondition R
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    f : LinearMap (RingHom.id R) (Finsupp α R) (Finsupp β R)
    i : Function.Surjective ⇑f
    P : LinearEquiv (RingHom.id R) (Finsupp β R) (β → R) := Finsupp.linearEquivFun …
    ⊢ LE.le (Fintype.card β) (Fintype.card α)
  -/
  let Q := (Finsupp.linearEquivFunOnFinite R R α).symm
  exact
    card_le_of_surjective R ((P.toLinearMap.comp f).comp Q.toLinearMap)
      ((P.surjective.comp i).comp Q.surjective)


/-- By the universal property for free modules, any surjective map `(Fin n → R) →ₗ[R] (Fin m → R)`
has an injective splitting `(Fin m → R) →ₗ[R] (Fin n → R)`
from which the strong rank condition gives the necessary inequality for the rank condition.
-/
instance (priority := 100) rankCondition_of_strongRankCondition [StrongRankCondition R] :
    RankCondition R where
  le_of_fin_surjective f s :=
    le_of_fin_injective R _ (f.splittingOfFunOnFintypeSurjective_injective s)


/-- We say that `R` has the invariant basis number property if `(Fin n → R) ≃ₗ[R] (Fin m → R)`
    implies `n = m`. This gives rise to a well-defined notion of rank of a finitely generated free
    module. -/
class InvariantBasisNumber : Prop where
  /-- Any linear equiv between `Rⁿ` and `Rᵐ` guarantees `m = n`. -/
  eq_of_fin_equiv : ∀ {n m : ℕ}, ((Fin n → R) ≃ₗ[R] Fin m → R) → n = m


instance (priority := 100) invariantBasisNumber_of_rankCondition [RankCondition R] :
    InvariantBasisNumber R where
  eq_of_fin_equiv e := le_antisymm (le_of_fin_surjective R e.symm.toLinearMap e.symm.surjective)
    (le_of_fin_surjective R e.toLinearMap e.surjective)


theorem eq_of_fin_equiv {n m : ℕ} : ((Fin n → R) ≃ₗ[R] Fin m → R) → n = m :=
  InvariantBasisNumber.eq_of_fin_equiv


theorem card_eq_of_linearEquiv {α β : Type*} [Fintype α] [Fintype β] (f : (α → R) ≃ₗ[R] β → R) :
    Fintype.card α = Fintype.card β :=
  eq_of_fin_equiv R
    ((LinearEquiv.funCongrLeft R R (Fintype.equivFin α)).trans f ≪≫ₗ
      (LinearEquiv.funCongrLeft R R (Fintype.equivFin β)).symm)
-- Porting note: this was not well-named because `lequiv` could mean other things
-- (e.g., `localEquiv`)


theorem nontrivial_of_invariantBasisNumber : Nontrivial R := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : InvariantBasisNumber R
    ⊢ Nontrivial R
  -/
  by_contra h
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : InvariantBasisNumber R
    h : Not (Nontrivial R)
    ⊢ False
  -/
  refine zero_ne_one (eq_of_fin_equiv R ?_)
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : InvariantBasisNumber R
    h : Not (Nontrivial R)
    ⊢ LinearEquiv (RingHom.id R) (Fin 0 → R) (Fin 1 → R)
  -/
  haveI := not_nontrivial_iff_subsingleton.1 h
  haveI : Subsingleton (Fin 1 → R) :=
    Subsingleton.intro fun a b => funext fun x => Subsingleton.elim _ _
  exact
    { toFun := 0
      invFun := 0
      map_add' := by aesop
      map_smul' := by aesop
      left_inv := fun _ => by simp [eq_iff_true_of_subsingleton]
      right_inv := fun _ => by simp [eq_iff_true_of_subsingleton] }


/-- Any nontrivial noetherian ring satisfies the strong rank condition,
    since it satisfies Orzech property. -/
instance (priority := 100) IsNoetherianRing.strongRankCondition : StrongRankCondition R :=
  inferInstance


/-- An `R`-linear map `R^n → R^m` induces a function `R^n/I^n → R^m/I^m`. -/
private def induced_map (I : Ideal R) (e : (ι → R) →ₗ[R] ι' → R) :
    (ι → R) ⧸ I.pi ι → (ι' → R) ⧸ I.pi ι' := fun x =>
  Quotient.liftOn' x (fun y => Ideal.Quotient.mk (I.pi ι') (e y))
    (by
      /-
        R : Type u
        inst✝¹ : CommRing R
        I✝ : Ideal R
        ι : Type v
        inst✝ : Fintype ι
        ι' : Type w
        I : Ideal R
        e : LinearMap (RingHom.id R) (ι → R) (ι' → R)
        x : HasQuotient.Quotient (ι → R) (I.pi ι)
        ⊢ ∀ (a b : ι → R), (Submodule.quotientRel (I.pi ι)) a b → Eq ((fun y => (Ideal …
      -/
      refine fun a b hab => Ideal.Quotient.eq.2 fun h => ?_
      /-
        R : Type u
        inst✝¹ : CommRing R
        I✝ : Ideal R
        ι : Type v
        inst✝ : Fintype ι
        ι' : Type w
        I : Ideal R
        e : LinearMap (RingHom.id R) (ι → R) (ι' → R)
        x : HasQuotient.Quotient (ι → R) (I.pi ι)
        a b : ι → R
        hab : (Submodule.quotientRel (I.pi ι)) a b
        h : ι'
        ⊢ Membership.mem I (HSub.hSub (e a) (e b) h)
      -/
      rw [Submodule.quotientRel_def] at hab
      /-
        R : Type u
        inst✝¹ : CommRing R
        I✝ : Ideal R
        ι : Type v
        inst✝ : Fintype ι
        ι' : Type w
        I : Ideal R
        e : LinearMap (RingHom.id R) (ι → R) (ι' → R)
        x : HasQuotient.Quotient (ι → R) (I.pi ι)
        a b : ι → R
        hab : Membership.mem (I.pi ι) (HSub.hSub a b)
        h : ι'
        ⊢ Membership.mem I (HSub.hSub (e a) (e b) h)
      -/
      rw [← LinearMap.map_sub]
      /-
        R : Type u
        inst✝¹ : CommRing R
        I✝ : Ideal R
        ι : Type v
        inst✝ : Fintype ι
        ι' : Type w
        I : Ideal R
        e : LinearMap (RingHom.id R) (ι → R) (ι' → R)
        x : HasQuotient.Quotient (ι → R) (I.pi ι)
        a b : ι → R
        hab : Membership.mem (I.pi ι) (HSub.hSub a b)
        h : ι'
        ⊢ Membership.mem I (e (HSub.hSub a b) h)
      -/
      exact Ideal.map_pi _ _ hab e h)
      /-
        🎉 no goals
      -/


/-- An isomorphism of `R`-modules `R^n ≃ R^m` induces an isomorphism of `R/I`-modules
    `R^n/I^n ≃ R^m/I^m`. -/
private def induced_equiv [Fintype ι'] (I : Ideal R) (e : (ι → R) ≃ₗ[R] ι' → R) :
    ((ι → R) ⧸ I.pi ι) ≃ₗ[R ⧸ I] (ι' → R) ⧸ I.pi ι' where
  -- Porting note: Lean couldn't correctly infer `(I.pi ι)` and `(I.pi ι')` on their own
  toFun := induced_map I e
  invFun := induced_map I e.symm
  map_add' := by
    /-
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      ⊢ ∀ (x y : HasQuotient.Quotient (ι → R) (I.pi ι)), Eq (induced_map I (↑e) (HAd …
    -/
    rintro ⟨a⟩ ⟨b⟩
    /-
      case mk.mk
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      y✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      ⊢ Eq (induced_map I (↑e) (HAdd.hAdd (Quot.mk (⇑(Submodule.quotientRel (I.pi ι) …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι') _ = Ideal.Quotient.mk (I.pi ι') _
    /-
      case mk.mk.convert_3
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      y✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι')) (↑e ((fun x1 x2 => HAdd.hAdd x1 x2) a b))) …
    -/
    congr
    /-
      case mk.mk.convert_3.h.e_6.h
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      y✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      ⊢ Eq (↑e ((fun x1 x2 => HAdd.hAdd x1 x2) a b)) ((fun x1 x2 => HAdd.hAdd x1 x2) …
    -/
    simp only [map_add]
    /-
      🎉 no goals
    -/
  map_smul' := by
    /-
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      ⊢ ∀ (m : HasQuotient.Quotient R I) (x : HasQuotient.Quotient (ι → R) (I.pi ι)) …
    -/
    rintro ⟨a⟩ ⟨b⟩
    /-
      case mk.mk
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      m✝ : HasQuotient.Quotient R I
      a : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      ⊢ Eq ({ toFun := induced_map I ↑e, map_add' := ⋯ }.toFun (HSMul.hSMul (Quot.mk …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι') _ = Ideal.Quotient.mk (I.pi ι') _
    /-
      case mk.mk.convert_3
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      m✝ : HasQuotient.Quotient R I
      a : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι')) (↑e (HSMul.hSMul a b))) ((Ideal.Quotient.m …
    -/
    congr
    /-
      case mk.mk.convert_3.h.e_6.h
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      m✝ : HasQuotient.Quotient R I
      a : R
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      b : ι → R
      ⊢ Eq (↑e (HSMul.hSMul a b)) (HSMul.hSMul a (↑e b))
    -/
    simp only [LinearEquiv.coe_coe, LinearEquiv.map_smulₛₗ, RingHom.id_apply]
    /-
      🎉 no goals
    -/
  left_inv := by
    /-
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      ⊢ Function.LeftInverse (induced_map I ↑e.symm) { toFun := induced_map I ↑e, ma …
    -/
    rintro ⟨a⟩
    /-
      case mk
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq (induced_map I (↑e.symm) ({ toFun := induced_map I ↑e, map_add' := ⋯, map …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι) _ = Ideal.Quotient.mk (I.pi ι) _
    /-
      case mk.convert_3
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι)) (↑e.symm (↑e a))) ((Ideal.Quotient.mk (I.pi …
    -/
    congr
    /-
      case mk.convert_3.h.e_6.h
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι → R) (I.pi ι)
      a : ι → R
      ⊢ Eq (↑e.symm (↑e a)) a
    -/
    simp only [LinearEquiv.coe_coe, LinearEquiv.symm_apply_apply]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      ⊢ Function.RightInverse (induced_map I ↑e.symm) { toFun := induced_map I ↑e, m …
    -/
    rintro ⟨a⟩
    /-
      case mk
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι' → R) (I.pi ι')
      a : ι' → R
      ⊢ Eq ({ toFun := induced_map I ↑e, map_add' := ⋯, map_smul' := ⋯ }.toFun (indu …
    -/
    convert_to Ideal.Quotient.mk (I.pi ι') _ = Ideal.Quotient.mk (I.pi ι') _
    /-
      case mk.convert_3
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι' → R) (I.pi ι')
      a : ι' → R
      ⊢ Eq ((Ideal.Quotient.mk (I.pi ι')) (↑e (↑e.symm a))) ((Ideal.Quotient.mk (I.p …
    -/
    congr
    /-
      case mk.convert_3.h.e_6.h
      R : Type u
      inst✝² : CommRing R
      I✝ : Ideal R
      ι : Type v
      inst✝¹ : Fintype ι
      ι' : Type w
      inst✝ : Fintype ι'
      I : Ideal R
      e : LinearEquiv (RingHom.id R) (ι → R) (ι' → R)
      x✝ : HasQuotient.Quotient (ι' → R) (I.pi ι')
      a : ι' → R
      ⊢ Eq (↑e (↑e.symm a)) a
    -/
    simp only [LinearEquiv.coe_coe,  LinearEquiv.apply_symm_apply]
    /-
      🎉 no goals
    -/


/-- Nontrivial commutative rings have the invariant basis number property.

In fact, any nontrivial commutative ring satisfies the strong rank condition, see
`commRing_strongRankCondition`. We prove this instance separately to avoid dependency on
`LinearAlgebra.Charpoly.Basic`. -/
instance (priority := 100) invariantBasisNumber_of_nontrivial_of_commRing {R : Type u} [CommRing R]
    [Nontrivial R] : InvariantBasisNumber R :=
  ⟨fun e =>
    let ⟨I, _hI⟩ := Ideal.exists_maximal R
    eq_of_fin_equiv (R ⧸ I)
      ((Ideal.piQuotEquiv _ _).symm ≪≫ₗ (induced_equiv _ e ≪≫ₗ Ideal.piQuotEquiv _ _))⟩


