/-- An R-module is projective if it is a direct summand of a free module, or equivalently
  if maps from the module lift along surjections. There are several other equivalent
  definitions. -/
class Module.Projective (R : Type*) [Semiring R] (P : Type*) [AddCommMonoid P] [Module R P] :
    Prop where
  out : ∃ s : P →ₗ[R] P →₀ R, Function.LeftInverse (Finsupp.linearCombination R id) s


theorem projective_def :
    Projective R P ↔ ∃ s : P →ₗ[R] P →₀ R, Function.LeftInverse (linearCombination R id) s :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem projective_def' :
    Projective R P ↔ ∃ s : P →ₗ[R] P →₀ R, Finsupp.linearCombination R id ∘ₗ s = .id := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    P : Type u_2
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    ⊢ Iff (Module.Projective R P) (Exists fun s => Eq ((Finsupp.linearCombination  …
  -/
  simp_rw [projective_def, DFunLike.ext_iff, Function.LeftInverse, comp_apply, id_apply]
  /-
    🎉 no goals
  -/


/-- A projective R-module has the property that maps from it lift along surjections. -/
theorem projective_lifting_property [h : Projective R P] (f : M →ₗ[R] N) (g : P →ₗ[R] N)
    (hf : Function.Surjective f) : ∃ h : P →ₗ[R] M, f ∘ₗ h = g := by
  /-
    Here's the first step of the proof.
    Recall that `X →₀ R` is Lean's way of talking about the free `R`-module
    on a type `X`. The universal property `Finsupp.linearCombination` says that to a map
    `X → N` from a type to an `R`-module, we get an associated R-module map
    `(X →₀ R) →ₗ N`. Apply this to a (noncomputable) map `P → M` coming from the map
    `P →ₗ N` and a random splitting of the surjection `M →ₗ N`, and we get
    a map `φ : (P →₀ R) →ₗ M`.
    -/
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    P : Type u_2
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R P
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    h : Module.Projective R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) P N
    hf : Function.Surjective ⇑f
    ⊢ Exists fun h => Eq (f.comp h) g
  -/
  let φ : (P →₀ R) →ₗ[R] M := Finsupp.linearCombination _ fun p => Function.surjInv hf (g p)
  -- By projectivity we have a map `P →ₗ (P →₀ R)`;
  /-
    R : Type u_1
    inst✝⁶ : Semiring R
    P : Type u_2
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R P
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    h : Module.Projective R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) P N
    hf : Function.Surjective ⇑f
    φ : LinearMap (RingHom.id R) (Finsupp P R) M := Finsupp.linearCombination R fu …
    ⊢ Exists fun h => Eq (f.comp h) g
  -/
  cases' h.out with s hs
  -- Compose to get `P →ₗ M`. This works.
  /-
    case intro
    R : Type u_1
    inst✝⁶ : Semiring R
    P : Type u_2
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R P
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    h : Module.Projective R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) P N
    hf : Function.Surjective ⇑f
    φ : LinearMap (RingHom.id R) (Finsupp P R) M := Finsupp.linearCombination R fu …
    s : LinearMap (RingHom.id R) P (Finsupp P R)
    hs : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑s
    ⊢ Exists fun h => Eq (f.comp h) g
  -/
  use φ.comp s
  /-
    case h
    R : Type u_1
    inst✝⁶ : Semiring R
    P : Type u_2
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R P
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    h : Module.Projective R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) P N
    hf : Function.Surjective ⇑f
    φ : LinearMap (RingHom.id R) (Finsupp P R) M := Finsupp.linearCombination R fu …
    s : LinearMap (RingHom.id R) P (Finsupp P R)
    hs : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑s
    ⊢ Eq (f.comp (φ.comp s)) g
  -/
  ext p
  /-
    case h.h
    R : Type u_1
    inst✝⁶ : Semiring R
    P : Type u_2
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R P
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    h : Module.Projective R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) P N
    hf : Function.Surjective ⇑f
    φ : LinearMap (RingHom.id R) (Finsupp P R) M := Finsupp.linearCombination R fu …
    s : LinearMap (RingHom.id R) P (Finsupp P R)
    hs : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑s
    p : P
    ⊢ Eq ((f.comp (φ.comp s)) p) (g p)
  -/
  conv_rhs => rw [← hs p]
  /-
    case h.h
    R : Type u_1
    inst✝⁶ : Semiring R
    P : Type u_2
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : Module R P
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_4
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    h : Module.Projective R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) P N
    hf : Function.Surjective ⇑f
    φ : LinearMap (RingHom.id R) (Finsupp P R) M := Finsupp.linearCombination R fu …
    s : LinearMap (RingHom.id R) P (Finsupp P R)
    hs : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑s
    p : P
    ⊢ Eq ((f.comp (φ.comp s)) p) (g ((Finsupp.linearCombination R id) (s p)))
  -/
  simp [φ, Finsupp.linearCombination_apply, Function.surjInv_eq hf, map_finsupp_sum]
  /-
    🎉 no goals
  -/


theorem _root_.LinearMap.exists_rightInverse_of_surjective [Projective R P]
    (f : M →ₗ[R] P) (hf_surj : range f = ⊤) : ∃ g : P →ₗ[R] M, f ∘ₗ g = LinearMap.id :=
  projective_lifting_property f (.id : P →ₗ[R] P) (LinearMap.range_eq_top.1 hf_surj)


/-- A module which satisfies the universal property is projective: If all surjections of
`R`-modules `(P →₀ R) →ₗ[R] P` have `R`-linear left inverse maps, then `P` is
projective. -/
theorem Projective.of_lifting_property'' {R : Type u} [Semiring R] {P : Type v} [AddCommMonoid P]
    [Module R P] (huniv : ∀ (f : (P →₀ R) →ₗ[R] P), Function.Surjective f →
      ∃ h : P →ₗ[R] (P →₀ R), f.comp h = .id) :
    Projective R P :=
  projective_def'.2 <| huniv (Finsupp.linearCombination R (id : P → P))
    (linearCombination_surjective _ Function.surjective_id)


instance [Projective R P] [Projective R Q] : Projective R (P × Q) := by
  /-
    R : Type u_1
    inst✝¹⁰ : Semiring R
    P : Type u_2
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    M : Type u_3
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_4
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    Q : Type u_5
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module.Projective R P
    inst✝ : Module.Projective R Q
    ⊢ Module.Projective R (Prod P Q)
  -/
  refine .of_lifting_property'' fun f hf ↦ ?_
  /-
    R : Type u_1
    inst✝¹⁰ : Semiring R
    P : Type u_2
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    M : Type u_3
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_4
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    Q : Type u_5
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module.Projective R P
    inst✝ : Module.Projective R Q
    f : LinearMap (RingHom.id R) (Finsupp (Prod P Q) R) (Prod P Q)
    hf : Function.Surjective ⇑f
    ⊢ Exists fun h => Eq (f.comp h) LinearMap.id
  -/
  rcases projective_lifting_property f (.inl _ _ _) hf with ⟨g₁, hg₁⟩
  /-
    case intro
    R : Type u_1
    inst✝¹⁰ : Semiring R
    P : Type u_2
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    M : Type u_3
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_4
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    Q : Type u_5
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module.Projective R P
    inst✝ : Module.Projective R Q
    f : LinearMap (RingHom.id R) (Finsupp (Prod P Q) R) (Prod P Q)
    hf : Function.Surjective ⇑f
    g₁ : LinearMap (RingHom.id R) P (Finsupp (Prod P Q) R)
    hg₁ : Eq (f.comp g₁) (LinearMap.inl R P Q)
    ⊢ Exists fun h => Eq (f.comp h) LinearMap.id
  -/
  rcases projective_lifting_property f (.inr _ _ _) hf with ⟨g₂, hg₂⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹⁰ : Semiring R
    P : Type u_2
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    M : Type u_3
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_4
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    Q : Type u_5
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module.Projective R P
    inst✝ : Module.Projective R Q
    f : LinearMap (RingHom.id R) (Finsupp (Prod P Q) R) (Prod P Q)
    hf : Function.Surjective ⇑f
    g₁ : LinearMap (RingHom.id R) P (Finsupp (Prod P Q) R)
    hg₁ : Eq (f.comp g₁) (LinearMap.inl R P Q)
    g₂ : LinearMap (RingHom.id R) Q (Finsupp (Prod P Q) R)
    hg₂ : Eq (f.comp g₂) (LinearMap.inr R P Q)
    ⊢ Exists fun h => Eq (f.comp h) LinearMap.id
  -/
  refine ⟨coprod g₁ g₂, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹⁰ : Semiring R
    P : Type u_2
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    M : Type u_3
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    N : Type u_4
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    Q : Type u_5
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module.Projective R P
    inst✝ : Module.Projective R Q
    f : LinearMap (RingHom.id R) (Finsupp (Prod P Q) R) (Prod P Q)
    hf : Function.Surjective ⇑f
    g₁ : LinearMap (RingHom.id R) P (Finsupp (Prod P Q) R)
    hg₁ : Eq (f.comp g₁) (LinearMap.inl R P Q)
    g₂ : LinearMap (RingHom.id R) Q (Finsupp (Prod P Q) R)
    hg₂ : Eq (f.comp g₂) (LinearMap.inr R P Q)
    ⊢ Eq (f.comp (g₁.coprod g₂)) LinearMap.id
  -/
  rw [LinearMap.comp_coprod, hg₁, hg₂, LinearMap.coprod_inl_inr]
  /-
    🎉 no goals
  -/


instance [h : ∀ i : ι, Projective R (A i)] : Projective R (Π₀ i, A i) :=
  .of_lifting_property'' fun f hf ↦ by
    classical
      choose g hg using fun i ↦ projective_lifting_property f (DFinsupp.lsingle i) hf
      replace hg : ∀ i x, f (g i x) = DFinsupp.single i x := fun i ↦ DFunLike.congr_fun (hg i)
      refine ⟨DFinsupp.coprodMap g, ?_⟩
      ext i x j
      simp only [comp_apply, id_apply, DFinsupp.lsingle_apply, DFinsupp.coprodMap_apply_single, hg]


/-- Free modules are projective. -/
theorem Projective.of_basis {ι : Type*} (b : Basis ι R P) : Projective R P := by
  -- need P →ₗ (P →₀ R) for definition of projective.
  -- get it from `ι → (P →₀ R)` coming from `b`.
  /-
    R : Type u_1
    inst✝² : Semiring R
    P : Type u_2
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    ι : Type u_8
    b : Basis ι R P
    ⊢ Module.Projective R P
  -/
  use b.constr ℕ fun i => Finsupp.single (b i) (1 : R)
  /-
    case h
    R : Type u_1
    inst✝² : Semiring R
    P : Type u_2
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    ι : Type u_8
    b : Basis ι R P
    ⊢ Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑((b.constr Nat) fun  …
  -/
  intro m
  simp only [b.constr_apply, mul_one, id, Finsupp.smul_single', Finsupp.linearCombination_single,
    map_finsupp_sum]
  /-
    case h
    R : Type u_1
    inst✝² : Semiring R
    P : Type u_2
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    ι : Type u_8
    b : Basis ι R P
    m : P
    ⊢ Eq ((b.repr m).sum fun a b_1 => HSMul.hSMul b_1 (b a)) m
  -/
  exact b.linearCombination_repr m
  /-
    🎉 no goals
  -/


instance (priority := 100) Projective.of_free [Module.Free R P] : Module.Projective R P :=
  .of_basis <| Module.Free.chooseBasis R P


/-- A direct summand of a projective module is projective. -/
theorem Projective.of_split [Module.Projective R M]
    (i : P →ₗ[R] M) (s : M →ₗ[R] P) (H : s.comp i = LinearMap.id) : Module.Projective R P := by
  obtain ⟨g, hg⟩ := projective_lifting_property (Finsupp.linearCombination R id) s
    (fun x ↦ ⟨Finsupp.single x 1, by simp⟩)
  /-
    case intro
    R : Type u_1
    inst✝⁵ : Semiring R
    P : Type u_2
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R P
    M : Type u_3
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Module.Projective R M
    i : LinearMap (RingHom.id R) P M
    s : LinearMap (RingHom.id R) M P
    H : Eq (s.comp i) LinearMap.id
    g : LinearMap (RingHom.id R) M (Finsupp P R)
    hg : Eq ((Finsupp.linearCombination R id).comp g) s
    ⊢ Module.Projective R P
  -/
  refine ⟨g.comp i, fun x ↦ ?_⟩
  rw [LinearMap.comp_apply, ← LinearMap.comp_apply, hg,
    ← LinearMap.comp_apply, H, LinearMap.id_apply]


theorem Projective.of_equiv [Module.Projective R M]
    (e : M ≃ₗ[R] P) : Module.Projective R P :=
                                               /-
                                                 R : Type u_1
                                                 inst✝⁵ : Semiring R
                                                 P : Type u_2
                                                 inst✝⁴ : AddCommMonoid P
                                                 inst✝³ : Module R P
                                                 M : Type u_3
                                                 inst✝² : AddCommMonoid M
                                                 inst✝¹ : Module R M
                                                 inst✝ : Module.Projective R M
                                                 e : LinearEquiv (RingHom.id R) M P
                                                 ⊢ Eq ((↑e).comp ↑e.symm) LinearMap.id
                                               -/
  Projective.of_split e.symm e.toLinearMap (by ext; simp)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- A quotient of a projective module is projective iff it is a direct summand. -/
theorem Projective.iff_split_of_projective [Module.Projective R M] (s : M →ₗ[R] P)
    (hs : Function.Surjective s) :
    Module.Projective R P ↔ ∃ i, s ∘ₗ i = LinearMap.id :=
  ⟨fun _ ↦ projective_lifting_property _ _ hs, fun ⟨i, H⟩ ↦ Projective.of_split i s H⟩


attribute [local instance] RingHomInvPair.of_ringEquiv in
theorem Projective.of_ringEquiv {R S} [Semiring R] [Semiring S] {M N}
    [AddCommMonoid M] [AddCommMonoid N] [Module R M] [Module S N]
    (e₁ : R ≃+* S) (e₂ : M ≃ₛₗ[RingHomClass.toRingHom e₁] N)
    [Projective R M] : Projective S N := by
  /-
    R : Type u_8
    S : Type u_9
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    M : Type u_10
    N : Type u_11
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module S N
    e₁ : RingEquiv R S
    e₂ : LinearEquiv (↑e₁) M N
    inst✝ : Module.Projective R M
    ⊢ Module.Projective S N
  -/
  obtain ⟨f, hf⟩ := ‹Projective R M›
  let g : N →ₗ[S] N →₀ S :=
  { toFun := fun x ↦ (equivCongrLeft e₂ (f (e₂.symm x))).mapRange e₁ e₁.map_zero
    map_add' := fun x y ↦ by ext; simp
    map_smul' := fun r v ↦ by ext i; simp [e₂.symm.map_smulₛₗ] }
  /-
    case mk.intro
    R : Type u_8
    S : Type u_9
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    M : Type u_10
    N : Type u_11
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module S N
    e₁ : RingEquiv R S
    e₂ : LinearEquiv (↑e₁) M N
    inst✝ : Module.Projective R M
    f : LinearMap (RingHom.id R) M (Finsupp M R)
    hf : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑f
    g : LinearMap (RingHom.id S) N (Finsupp N S) := { toFun := fun x => Finsupp.ma …
    ⊢ Module.Projective S N
  -/
  refine ⟨⟨g, fun x ↦ ?_⟩⟩
  /-
    case mk.intro
    R : Type u_8
    S : Type u_9
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    M : Type u_10
    N : Type u_11
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module S N
    e₁ : RingEquiv R S
    e₂ : LinearEquiv (↑e₁) M N
    inst✝ : Module.Projective R M
    f : LinearMap (RingHom.id R) M (Finsupp M R)
    hf : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑f
    g : LinearMap (RingHom.id S) N (Finsupp N S) := { toFun := fun x => Finsupp.ma …
    x : N
    ⊢ Eq ((Finsupp.linearCombination S id) (g x)) x
  -/
  replace hf := congr(e₂ $(hf (e₂.symm x)))
  /-
    case mk.intro
    R : Type u_8
    S : Type u_9
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring S
    M : Type u_10
    N : Type u_11
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module S N
    e₁ : RingEquiv R S
    e₂ : LinearEquiv (↑e₁) M N
    inst✝ : Module.Projective R M
    f : LinearMap (RingHom.id R) M (Finsupp M R)
    g : LinearMap (RingHom.id S) N (Finsupp N S) := { toFun := fun x => Finsupp.ma …
    x : N
    hf : Eq (e₂ ((Finsupp.linearCombination R id) (f (e₂.symm x)))) (e₂ (e₂.symm x))
    ⊢ Eq ((Finsupp.linearCombination S id) (g x)) x
  -/
  simpa [linearCombination_apply, sum_mapRange_index, g, map_finsupp_sum, e₂.map_smulₛₗ] using hf
  /-
    🎉 no goals
  -/


/-- A module is projective iff it is the direct summand of a free module. -/
theorem Projective.iff_split : Module.Projective R P ↔
    ∃ (M : Type max u v) (_ : AddCommMonoid M) (_ : Module R M) (_ : Module.Free R M)
      (i : P →ₗ[R] M) (s : M →ₗ[R] P), s.comp i = LinearMap.id :=
  ⟨fun ⟨i, hi⟩ ↦ ⟨P →₀ R, _, _, inferInstance, i, Finsupp.linearCombination R id, LinearMap.ext hi⟩,
    fun ⟨_, _, _, _, i, s, H⟩ ↦ Projective.of_split i s H⟩


set_option maxSynthPendingDepth 2 in
open TensorProduct in
instance Projective.tensorProduct [hM : Module.Projective R M] [hN : Module.Projective R₀ N] :
    Module.Projective R (M ⊗[R₀] N) := by
  /-
    R : Type u
    inst✝¹⁰ : Semiring R
    P : Type v
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    R₀ : Type u_2
    M : Type u_1
    N : Type u_3
    inst✝⁷ : CommSemiring R₀
    inst✝⁶ : Algebra R₀ R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R₀ M
    inst✝³ : Module R M
    inst✝² : IsScalarTower R₀ R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R₀ N
    hM : Module.Projective R M
    hN : Module.Projective R₀ N
    ⊢ Module.Projective R (TensorProduct R₀ M N)
  -/
  obtain ⟨sM, hsM⟩ := hM
  /-
    case mk.intro
    R : Type u
    inst✝¹⁰ : Semiring R
    P : Type v
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    R₀ : Type u_2
    M : Type u_1
    N : Type u_3
    inst✝⁷ : CommSemiring R₀
    inst✝⁶ : Algebra R₀ R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R₀ M
    inst✝³ : Module R M
    inst✝² : IsScalarTower R₀ R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R₀ N
    hN : Module.Projective R₀ N
    sM : LinearMap (RingHom.id R) M (Finsupp M R)
    hsM : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑sM
    ⊢ Module.Projective R (TensorProduct R₀ M N)
  -/
  obtain ⟨sN, hsN⟩ := hN
  have : Module.Projective R (M ⊗[R₀] (N →₀ R₀)) := by
    fapply Projective.of_split (R := R) (M := ((M →₀ R) ⊗[R₀] (N →₀ R₀)))
    · exact (AlgebraTensorModule.map sM (LinearMap.id (R := R₀) (M := N →₀ R₀)))
    · exact (AlgebraTensorModule.map
        (Finsupp.linearCombination R id) (LinearMap.id (R := R₀) (M := N →₀ R₀)))
    · ext; simp [hsM _]
  /-
    case mk.intro.mk.intro
    R : Type u
    inst✝¹⁰ : Semiring R
    P : Type v
    inst✝⁹ : AddCommMonoid P
    inst✝⁸ : Module R P
    R₀ : Type u_2
    M : Type u_1
    N : Type u_3
    inst✝⁷ : CommSemiring R₀
    inst✝⁶ : Algebra R₀ R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R₀ M
    inst✝³ : Module R M
    inst✝² : IsScalarTower R₀ R M
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R₀ N
    sM : LinearMap (RingHom.id R) M (Finsupp M R)
    hsM : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑sM
    sN : LinearMap (RingHom.id R₀) N (Finsupp N R₀)
    hsN : Function.LeftInverse ⇑(Finsupp.linearCombination R₀ id) ⇑sN
    this : Module.Projective R (TensorProduct R₀ M (Finsupp N R₀))
    ⊢ Module.Projective R (TensorProduct R₀ M N)
  -/
  fapply Projective.of_split (R := R) (M := (M ⊗[R₀] (N →₀ R₀)))
    /-
      case mk.intro.mk.intro.i
      R : Type u
      inst✝¹⁰ : Semiring R
      P : Type v
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      R₀ : Type u_2
      M : Type u_1
      N : Type u_3
      inst✝⁷ : CommSemiring R₀
      inst✝⁶ : Algebra R₀ R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R₀ M
      inst✝³ : Module R M
      inst✝² : IsScalarTower R₀ R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R₀ N
      sM : LinearMap (RingHom.id R) M (Finsupp M R)
      hsM : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑sM
      sN : LinearMap (RingHom.id R₀) N (Finsupp N R₀)
      hsN : Function.LeftInverse ⇑(Finsupp.linearCombination R₀ id) ⇑sN
      this : Module.Projective R (TensorProduct R₀ M (Finsupp N R₀))
      ⊢ LinearMap (RingHom.id R) (TensorProduct R₀ M N) (TensorProduct R₀ M (Finsupp …
    -/
  · exact (AlgebraTensorModule.map (LinearMap.id (R := R) (M := M)) sN)
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.mk.intro.s
      R : Type u
      inst✝¹⁰ : Semiring R
      P : Type v
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      R₀ : Type u_2
      M : Type u_1
      N : Type u_3
      inst✝⁷ : CommSemiring R₀
      inst✝⁶ : Algebra R₀ R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R₀ M
      inst✝³ : Module R M
      inst✝² : IsScalarTower R₀ R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R₀ N
      sM : LinearMap (RingHom.id R) M (Finsupp M R)
      hsM : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑sM
      sN : LinearMap (RingHom.id R₀) N (Finsupp N R₀)
      hsN : Function.LeftInverse ⇑(Finsupp.linearCombination R₀ id) ⇑sN
      this : Module.Projective R (TensorProduct R₀ M (Finsupp N R₀))
      ⊢ LinearMap (RingHom.id R) (TensorProduct R₀ M (Finsupp N R₀)) (TensorProduct  …
    -/
  · exact (AlgebraTensorModule.map (LinearMap.id (R := R) (M := M)) (linearCombination R₀ id))
    /-
      🎉 no goals
    -/
    /-
      case mk.intro.mk.intro.H
      R : Type u
      inst✝¹⁰ : Semiring R
      P : Type v
      inst✝⁹ : AddCommMonoid P
      inst✝⁸ : Module R P
      R₀ : Type u_2
      M : Type u_1
      N : Type u_3
      inst✝⁷ : CommSemiring R₀
      inst✝⁶ : Algebra R₀ R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R₀ M
      inst✝³ : Module R M
      inst✝² : IsScalarTower R₀ R M
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R₀ N
      sM : LinearMap (RingHom.id R) M (Finsupp M R)
      hsM : Function.LeftInverse ⇑(Finsupp.linearCombination R id) ⇑sM
      sN : LinearMap (RingHom.id R₀) N (Finsupp N R₀)
      hsN : Function.LeftInverse ⇑(Finsupp.linearCombination R₀ id) ⇑sN
      this : Module.Projective R (TensorProduct R₀ M (Finsupp N R₀))
      ⊢ Eq ((TensorProduct.AlgebraTensorModule.map LinearMap.id (Finsupp.linearCombi …
    -/
  · ext; simp [hsN _]
         /-
           🎉 no goals
         -/


/-- A module which satisfies the universal property is projective. Note that the universe variables
in `huniv` are somewhat restricted. -/
theorem Projective.of_lifting_property' {R : Type u} [Semiring R] {P : Type max u v}
    [AddCommMonoid P] [Module R P]
    -- If for all surjections of `R`-modules `M →ₗ N`, all maps `P →ₗ N` lift to `P →ₗ M`,
    (huniv : ∀ {M : Type max v u} {N : Type max u v} [AddCommMonoid M] [AddCommMonoid N]
      [Module R M] [Module R N] (f : M →ₗ[R] N) (g : P →ₗ[R] N),
        Function.Surjective f → ∃ h : P →ₗ[R] M, f.comp h = g) :
    -- then `P` is projective.
    Projective R P :=
  .of_lifting_property'' (huniv · _)

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: generalize to `P : Type v`?

/-- A variant of `of_lifting_property'` when we're working over a `[Ring R]`,
which only requires quantifying over modules with an `AddCommGroup` instance. -/
theorem Projective.of_lifting_property {R : Type u} [Ring R] {P : Type max u v} [AddCommGroup P]
    [Module R P]
    -- If for all surjections of `R`-modules `M →ₗ N`, all maps `P →ₗ N` lift to `P →ₗ M`,
    (huniv : ∀ {M : Type max v u} {N : Type max u v} [AddCommGroup M] [AddCommGroup N]
      [Module R M] [Module R N] (f : M →ₗ[R] N) (g : P →ₗ[R] N),
        Function.Surjective f → ∃ h : P →ₗ[R] M, f.comp h = g) :
    -- then `P` is projective.
    Projective R P :=
  .of_lifting_property'' (huniv · _)


