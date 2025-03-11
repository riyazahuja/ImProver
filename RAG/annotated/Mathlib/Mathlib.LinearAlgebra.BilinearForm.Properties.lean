/-- The proposition that a bilinear form is reflexive -/
def IsRefl (B : BilinForm R M) : Prop := LinearMap.IsRefl B


theorem eq_zero (H : B.IsRefl) : ∀ {x y : M}, B x y = 0 → B y x = 0 := fun {x y} => H x y


protected theorem neg {B : BilinForm R₁ M₁} (hB : B.IsRefl) : (-B).IsRefl := fun x y =>
  neg_eq_zero.mpr ∘ hB x y ∘ neg_eq_zero.mp


protected theorem smul {α} [CommSemiring α] [Module α R] [SMulCommClass R α R]
    [NoZeroSMulDivisors α R] (a : α) {B : BilinForm R M} (hB : B.IsRefl) :
    (a • B).IsRefl := fun _ _ h =>
  (smul_eq_zero.mp h).elim (fun ha => smul_eq_zero_of_left ha _) fun hBz =>
    smul_eq_zero_of_right _ (hB _ _ hBz)


protected theorem groupSMul {α} [Group α] [DistribMulAction α R] [SMulCommClass R α R] (a : α)
    {B : BilinForm R M} (hB : B.IsRefl) : (a • B).IsRefl := fun x y =>
  (smul_eq_zero_iff_eq _).mpr ∘ hB x y ∘ (smul_eq_zero_iff_eq _).mp


@[simp]
theorem isRefl_zero : (0 : BilinForm R M).IsRefl := fun _ _ _ => rfl


@[simp]
theorem isRefl_neg {B : BilinForm R₁ M₁} : (-B).IsRefl ↔ B.IsRefl :=
  ⟨fun h => neg_neg B ▸ h.neg, IsRefl.neg⟩


/-- The proposition that a bilinear form is symmetric -/
def IsSymm (B : BilinForm R M) : Prop := LinearMap.IsSymm B


protected theorem eq (H : B.IsSymm) (x y : M) : B x y = B y x :=
  H x y


theorem isRefl (H : B.IsSymm) : B.IsRefl := fun x y H1 => H x y ▸ H1


protected theorem add {B₁ B₂ : BilinForm R M} (hB₁ : B₁.IsSymm) (hB₂ : B₂.IsSymm) :
    (B₁ + B₂).IsSymm := fun x y => (congr_arg₂ (· + ·) (hB₁ x y) (hB₂ x y) : _)


protected theorem sub {B₁ B₂ : BilinForm R₁ M₁} (hB₁ : B₁.IsSymm) (hB₂ : B₂.IsSymm) :
    (B₁ - B₂).IsSymm := fun x y => (congr_arg₂ Sub.sub (hB₁ x y) (hB₂ x y) : _)


protected theorem neg {B : BilinForm R₁ M₁} (hB : B.IsSymm) : (-B).IsSymm := fun x y =>
  congr_arg Neg.neg (hB x y)


protected theorem smul {α} [Monoid α] [DistribMulAction α R] [SMulCommClass R α R] (a : α)
    {B : BilinForm R M} (hB : B.IsSymm) : (a • B).IsSymm := fun x y =>
  congr_arg (a • ·) (hB x y)


/-- The restriction of a symmetric bilinear form on a submodule is also symmetric. -/
theorem restrict {B : BilinForm R M} (b : B.IsSymm) (W : Submodule R M) :
    (B.restrict W).IsSymm := fun x y => b x y


@[simp]
theorem isSymm_zero : (0 : BilinForm R M).IsSymm := fun _ _ => rfl


@[simp]
theorem isSymm_neg {B : BilinForm R₁ M₁} : (-B).IsSymm ↔ B.IsSymm :=
  ⟨fun h => neg_neg B ▸ h.neg, IsSymm.neg⟩


theorem isSymm_iff_flip : B.IsSymm ↔ flipHom B = B :=
                               /-
                                 R : Type u_1
                                 M : Type u_2
                                 inst✝² : CommSemiring R
                                 inst✝¹ : AddCommMonoid M
                                 inst✝ : Module R M
                                 B : LinearMap.BilinForm R M
                                 x✝¹ x✝ : M
                                 ⊢ Iff (Eq ((RingHom.id R) ((B x✝¹) x✝)) ((B x✝) x✝¹)) (Eq (((LinearMap.BilinFo …
                               -/
  (forall₂_congr fun _ _ => by exact eq_comm).trans BilinForm.ext_iff.symm
                               /-
                                 🎉 no goals
                               -/


/-- The proposition that a bilinear form is alternating -/
def IsAlt (B : BilinForm R M) : Prop := LinearMap.IsAlt B


theorem self_eq_zero (H : B.IsAlt) (x : M) : B x x = 0 := LinearMap.IsAlt.self_eq_zero H x


theorem neg_eq (H : B₁.IsAlt) (x y : M₁) : -B₁ x y = B₁ y x := LinearMap.IsAlt.neg H x y


theorem isRefl (H : B₁.IsAlt) : B₁.IsRefl := LinearMap.IsAlt.isRefl H


theorem eq_of_add_add_eq_zero [IsCancelAdd R] {a b c : M} (H : B.IsAlt) (hAdd : a + b + c = 0) :
    B a b = B b c := LinearMap.IsAlt.eq_of_add_add_eq_zero H hAdd


protected theorem add {B₁ B₂ : BilinForm R M} (hB₁ : B₁.IsAlt) (hB₂ : B₂.IsAlt) : (B₁ + B₂).IsAlt :=
  fun x => (congr_arg₂ (· + ·) (hB₁ x) (hB₂ x) : _).trans <| add_zero _


protected theorem sub {B₁ B₂ : BilinForm R₁ M₁} (hB₁ : B₁.IsAlt) (hB₂ : B₂.IsAlt) :
    (B₁ - B₂).IsAlt := fun x => (congr_arg₂ Sub.sub (hB₁ x) (hB₂ x)).trans <| sub_zero _


protected theorem neg {B : BilinForm R₁ M₁} (hB : B.IsAlt) : (-B).IsAlt := fun x =>
  neg_eq_zero.mpr <| hB x


protected theorem smul {α} [Monoid α] [DistribMulAction α R] [SMulCommClass R α R] (a : α)
    {B : BilinForm R M} (hB : B.IsAlt) : (a • B).IsAlt := fun x =>
  (congr_arg (a • ·) (hB x)).trans <| smul_zero _


@[simp]
theorem isAlt_zero : (0 : BilinForm R M).IsAlt := fun _ => rfl


@[simp]
theorem isAlt_neg {B : BilinForm R₁ M₁} : (-B).IsAlt ↔ B.IsAlt :=
  ⟨fun h => neg_neg B ▸ h.neg, IsAlt.neg⟩


/-- A nondegenerate bilinear form is a bilinear form such that the only element that is orthogonal
to every other element is `0`; i.e., for all nonzero `m` in `M`, there exists `n` in `M` with
`B m n ≠ 0`.

Note that for general (neither symmetric nor antisymmetric) bilinear forms this definition has a
chirality; in addition to this "left" nondegeneracy condition one could define a "right"
nondegeneracy condition that in the situation described, `B n m ≠ 0`.  This variant definition is
not currently provided in mathlib. In finite dimension either definition implies the other. -/
def Nondegenerate (B : BilinForm R M) : Prop :=
  ∀ m : M, (∀ n : M, B m n = 0) → m = 0


/-- In a non-trivial module, zero is not non-degenerate. -/
theorem not_nondegenerate_zero [Nontrivial M] : ¬(0 : BilinForm R M).Nondegenerate :=
  let ⟨m, hm⟩ := exists_ne (0 : M)
  fun h => hm (h m fun _ => rfl)


theorem Nondegenerate.ne_zero [Nontrivial M] {B : BilinForm R M} (h : B.Nondegenerate) : B ≠ 0 :=
  fun h0 => not_nondegenerate_zero R M <| h0 ▸ h


theorem Nondegenerate.congr {B : BilinForm R M} (e : M ≃ₗ[R] M') (h : B.Nondegenerate) :
    (congr e B).Nondegenerate := fun m hm =>
  e.symm.map_eq_zero_iff.1 <|
    h (e.symm m) fun n => (congr_arg _ (e.symm_apply_apply n).symm).trans (hm (e n))


@[simp]
theorem nondegenerate_congr_iff {B : BilinForm R M} (e : M ≃ₗ[R] M') :
    (congr e B).Nondegenerate ↔ B.Nondegenerate :=
  ⟨fun h => by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_8
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B : LinearMap.BilinForm R M
      e : LinearEquiv (RingHom.id R) M M'
      h : ((LinearMap.BilinForm.congr e) B).Nondegenerate
      ⊢ B.Nondegenerate
    -/
    convert h.congr e.symm
    /-
      case h.e'_6
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      M' : Type u_8
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      B : LinearMap.BilinForm R M
      e : LinearEquiv (RingHom.id R) M M'
      h : ((LinearMap.BilinForm.congr e) B).Nondegenerate
      ⊢ Eq B ((LinearMap.BilinForm.congr e.symm) ((LinearMap.BilinForm.congr e) B))
    -/
    rw [congr_congr, e.self_trans_symm, congr_refl, LinearEquiv.refl_apply], Nondegenerate.congr e⟩
    /-
      🎉 no goals
    -/


/-- A bilinear form is nondegenerate if and only if it has a trivial kernel. -/
theorem nondegenerate_iff_ker_eq_bot {B : BilinForm R M} :
    B.Nondegenerate ↔ LinearMap.ker B = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ⊢ Iff B.Nondegenerate (Eq (LinearMap.ker B) Bot.bot)
  -/
  rw [LinearMap.ker_eq_bot']
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    ⊢ Iff B.Nondegenerate (∀ (m : M), Eq (B m) 0 → Eq m 0)
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : B.Nondegenerate
      ⊢ ∀ (m : M), Eq (B m) 0 → Eq m 0
    -/
  · refine fun m hm => h _ fun x => ?_
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : B.Nondegenerate
      m : M
      hm : Eq (B m) 0
      x : M
      ⊢ Eq ((B m) x) 0
    -/
    rw [hm]
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : B.Nondegenerate
      m : M
      hm : Eq (B m) 0
      x : M
      ⊢ Eq (0 x) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : ∀ (m : M), Eq (B m) 0 → Eq m 0
      ⊢ B.Nondegenerate
    -/
  · intro m hm
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : ∀ (m : M), Eq (B m) 0 → Eq m 0
      m : M
      hm : ∀ (n : M), Eq ((B m) n) 0
      ⊢ Eq m 0
    -/
    apply h
    /-
      case mpr.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : ∀ (m : M), Eq (B m) 0 → Eq m 0
      m : M
      hm : ∀ (n : M), Eq ((B m) n) 0
      ⊢ Eq (B m) 0
    -/
    ext x
    /-
      case mpr.a.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      B : LinearMap.BilinForm R M
      h : ∀ (m : M), Eq (B m) 0 → Eq m 0
      m : M
      hm : ∀ (n : M), Eq ((B m) n) 0
      x : M
      ⊢ Eq ((B m) x) (0 x)
    -/
    exact hm x
    /-
      🎉 no goals
    -/


theorem Nondegenerate.ker_eq_bot {B : BilinForm R M} (h : B.Nondegenerate) :
    LinearMap.ker B = ⊥ := nondegenerate_iff_ker_eq_bot.mp h


theorem compLeft_injective (B : BilinForm R₁ M₁) (b : B.Nondegenerate) :
    Function.Injective B.compLeft := fun φ ψ h => by
  /-
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.Nondegenerate
    φ ψ : LinearMap (RingHom.id R₁) M₁ M₁
    h : Eq (B.compLeft φ) (B.compLeft ψ)
    ⊢ Eq φ ψ
  -/
  ext w
  /-
    case h
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.Nondegenerate
    φ ψ : LinearMap (RingHom.id R₁) M₁ M₁
    h : Eq (B.compLeft φ) (B.compLeft ψ)
    w : M₁
    ⊢ Eq (φ w) (ψ w)
  -/
  refine eq_of_sub_eq_zero (b _ ?_)
  /-
    case h
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.Nondegenerate
    φ ψ : LinearMap (RingHom.id R₁) M₁ M₁
    h : Eq (B.compLeft φ) (B.compLeft ψ)
    w : M₁
    ⊢ ∀ (n : M₁), Eq ((B (HSub.hSub (φ w) (ψ w))) n) 0
  -/
  intro v
  /-
    case h
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.Nondegenerate
    φ ψ : LinearMap (RingHom.id R₁) M₁ M₁
    h : Eq (B.compLeft φ) (B.compLeft ψ)
    w v : M₁
    ⊢ Eq ((B (HSub.hSub (φ w) (ψ w))) v) 0
  -/
  rw [sub_left, ← compLeft_apply, ← compLeft_apply, ← h, sub_self]
  /-
    🎉 no goals
  -/


theorem isAdjointPair_unique_of_nondegenerate (B : BilinForm R₁ M₁) (b : B.Nondegenerate)
    (φ ψ₁ ψ₂ : M₁ →ₗ[R₁] M₁) (hψ₁ : IsAdjointPair B B ψ₁ φ) (hψ₂ : IsAdjointPair B B ψ₂ φ) :
    ψ₁ = ψ₂ :=
                                              /-
                                                R₁ : Type u_3
                                                M₁ : Type u_4
                                                inst✝² : CommRing R₁
                                                inst✝¹ : AddCommGroup M₁
                                                inst✝ : Module R₁ M₁
                                                B : LinearMap.BilinForm R₁ M₁
                                                b : B.Nondegenerate
                                                φ ψ₁ ψ₂ : LinearMap (RingHom.id R₁) M₁ M₁
                                                hψ₁ : LinearMap.IsAdjointPair B B ⇑ψ₁ ⇑φ
                                                hψ₂ : LinearMap.IsAdjointPair B B ⇑ψ₂ ⇑φ
                                                v w : M₁
                                                ⊢ Eq (((B.compLeft ψ₁) v) w) (((B.compLeft ψ₂) v) w)
                                              -/
  B.compLeft_injective b <| ext fun v w => by rw [compLeft_apply, compLeft_apply, hψ₁, hψ₂]
                                              /-
                                                🎉 no goals
                                              -/


/-- Given a nondegenerate bilinear form `B` on a finite-dimensional vector space, `B.toDual` is
the linear equivalence between a vector space and its dual. -/
noncomputable def toDual (B : BilinForm K V) (b : B.Nondegenerate) : V ≃ₗ[K] Module.Dual K V :=
  B.linearEquivOfInjective (LinearMap.ker_eq_bot.mp <| b.ker_eq_bot)
    Subspace.dual_finrank_eq.symm


theorem toDual_def {B : BilinForm K V} (b : B.SeparatingLeft) {m n : V} : B.toDual b m n = B m n :=
  rfl


@[simp]
lemma apply_toDual_symm_apply {B : BilinForm K V} {hB : B.Nondegenerate}
    (f : Module.Dual K V) (v : V) :
    B ((B.toDual hB).symm f) v = f v := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    f : Module.Dual K V
    v : V
    ⊢ Eq ((B ((B.toDual hB).symm f)) v) (f v)
  -/
  change B.toDual hB ((B.toDual hB).symm f) v = f v
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    f : Module.Dual K V
    v : V
    ⊢ Eq (((B.toDual hB) ((B.toDual hB).symm f)) v) (f v)
  -/
  simp only [LinearEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma Nondegenerate.flip {B : BilinForm K V} (hB : B.Nondegenerate) :
    B.flip.Nondegenerate := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    ⊢ B.flip.Nondegenerate
  -/
  intro x hx
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    x : V
    hx : ∀ (n : V), Eq ((B.flip x) n) 0
    ⊢ Eq x 0
  -/
  apply (Module.evalEquiv K V).injective
  /-
    case a
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    x : V
    hx : ∀ (n : V), Eq ((B.flip x) n) 0
    ⊢ Eq ((Module.evalEquiv K V) x) ((Module.evalEquiv K V) 0)
  -/
  ext f
  /-
    case a.h
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    x : V
    hx : ∀ (n : V), Eq ((B.flip x) n) 0
    f : Module.Dual K V
    ⊢ Eq (((Module.evalEquiv K V) x) f) (((Module.evalEquiv K V) 0) f)
  -/
  obtain ⟨y, rfl⟩ := (B.toDual hB).surjective f
  /-
    case a.h.intro
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    x : V
    hx : ∀ (n : V), Eq ((B.flip x) n) 0
    y : V
    ⊢ Eq (((Module.evalEquiv K V) x) ((B.toDual hB) y)) (((Module.evalEquiv K V) 0 …
  -/
  simpa using hx y
  /-
    🎉 no goals
  -/


lemma nonDegenerateFlip_iff {B : BilinForm K V} :
    B.flip.Nondegenerate ↔ B.Nondegenerate := ⟨Nondegenerate.flip, Nondegenerate.flip⟩


/-- The `B`-dual basis `B.dualBasis hB b` to a finite basis `b` satisfies
`B (B.dualBasis hB b i) (b j) = B (b i) (B.dualBasis hB b j) = if i = j then 1 else 0`,
where `B` is a nondegenerate (symmetric) bilinear form and `b` is a finite basis. -/
noncomputable def dualBasis (B : BilinForm K V) (hB : B.Nondegenerate) (b : Basis ι K V) :
    Basis ι K V :=
  haveI := FiniteDimensional.of_fintype_basis b
  b.dualBasis.map (B.toDual hB).symm


@[simp]
theorem dualBasis_repr_apply
    (B : BilinForm K V) (hB : B.Nondegenerate) (b : Basis ι K V) (x i) :
    (B.dualBasis hB b).repr x i = B x (b i) := by
  #adaptation_note
  /-- Before https://github.com/leanprover/lean4/pull/4814, we did not need the `@` in front of `toDual_def` in the `rw`.
  I'm confused! -/
  rw [dualBasis, Basis.map_repr, LinearEquiv.symm_symm, LinearEquiv.trans_apply,
    Basis.dualBasis_repr, @toDual_def]


theorem apply_dualBasis_left (B : BilinForm K V) (hB : B.Nondegenerate) (b : Basis ι K V) (i j) :
    B (B.dualBasis hB b i) (b j) = if j = i then 1 else 0 := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Type u_9
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    b : Basis ι K V
    i j : ι
    ⊢ Eq ((B ((B.dualBasis hB b) i)) (b j)) (ite (Eq j i) 1 0)
  -/
  have := FiniteDimensional.of_fintype_basis b
  rw [dualBasis, Basis.map_apply, Basis.coe_dualBasis, ← toDual_def hB,
    LinearEquiv.apply_symm_apply, Basis.coord_apply, Basis.repr_self, Finsupp.single_apply]


theorem apply_dualBasis_right (B : BilinForm K V) (hB : B.Nondegenerate) (sym : B.IsSymm)
    (b : Basis ι K V) (i j) : B (b i) (B.dualBasis hB b j) = if i = j then 1 else 0 := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝⁴ : Field K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Type u_9
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    sym : B.IsSymm
    b : Basis ι K V
    i j : ι
    ⊢ Eq ((B (b i)) ((B.dualBasis hB b) j)) (ite (Eq i j) 1 0)
  -/
  rw [sym.eq, apply_dualBasis_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma dualBasis_dualBasis_flip [FiniteDimensional K V]
    (B : BilinForm K V) (hB : B.Nondegenerate) {ι : Type*}
    [Finite ι] [DecidableEq ι] (b : Basis ι K V) :
    B.dualBasis hB (B.flip.dualBasis hB.flip b) = b := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝⁵ : Field K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    inst✝² : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    ι : Type u_10
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    b : Basis ι K V
    ⊢ Eq (B.dualBasis hB (B.flip.dualBasis ⋯ b)) b
  -/
  ext i
  /-
    case a
    V : Type u_5
    K : Type u_6
    inst✝⁵ : Field K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    inst✝² : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    ι : Type u_10
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    b : Basis ι K V
    i : ι
    ⊢ Eq ((B.dualBasis hB (B.flip.dualBasis ⋯ b)) i) (b i)
  -/
  refine LinearMap.ker_eq_bot.mp hB.ker_eq_bot ((B.flip.dualBasis hB.flip b).ext (fun j ↦ ?_))
  /-
    case a
    V : Type u_5
    K : Type u_6
    inst✝⁵ : Field K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    inst✝² : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    ι : Type u_10
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    b : Basis ι K V
    i j : ι
    ⊢ Eq ((B ((B.dualBasis hB (B.flip.dualBasis ⋯ b)) i)) ((B.flip.dualBasis ⋯ b)  …
  -/
  simp_rw [apply_dualBasis_left, ← B.flip_apply, apply_dualBasis_left, @eq_comm _ i j]
  /-
    🎉 no goals
  -/


@[simp]
lemma dualBasis_flip_dualBasis (B : BilinForm K V) (hB : B.Nondegenerate) {ι}
    [Finite ι] [DecidableEq ι] [FiniteDimensional K V] (b : Basis ι K V) :
    B.flip.dualBasis hB.flip (B.dualBasis hB b) = b :=
  dualBasis_dualBasis_flip _ hB.flip b


@[simp]
lemma dualBasis_dualBasis (B : BilinForm K V) (hB : B.Nondegenerate) (hB' : B.IsSymm) {ι}
    [Finite ι] [DecidableEq ι] [FiniteDimensional K V] (b : Basis ι K V) :
    B.dualBasis hB (B.dualBasis hB b) = b := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝⁵ : Field K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB' : B.IsSymm
    ι : Type u_10
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    inst✝ : FiniteDimensional K V
    b : Basis ι K V
    ⊢ Eq (B.dualBasis hB (B.dualBasis hB b)) b
  -/
  convert dualBasis_dualBasis_flip _ hB.flip b
  /-
    case h.e'_2.h.e'_9
    V : Type u_5
    K : Type u_6
    inst✝⁵ : Field K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB' : B.IsSymm
    ι : Type u_10
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    inst✝ : FiniteDimensional K V
    b : Basis ι K V
    ⊢ Eq B B.flip
  -/
  rwa [eq_comm, ← isSymm_iff_flip]
  /-
    🎉 no goals
  -/


/-- Given bilinear forms `B₁, B₂` where `B₂` is nondegenerate, `symmCompOfNondegenerate`
is the linear map `B₂ ∘ B₁`. -/
noncomputable def symmCompOfNondegenerate (B₁ B₂ : BilinForm K V) (b₂ : B₂.Nondegenerate) :
    V →ₗ[K] V :=
  (B₂.toDual b₂).symm.toLinearMap.comp B₁


theorem comp_symmCompOfNondegenerate_apply (B₁ : BilinForm K V) {B₂ : BilinForm K V}
    (b₂ : B₂.Nondegenerate) (v : V) :
    B₂ (B₁.symmCompOfNondegenerate B₂ b₂ v) = B₁ v := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B₁ B₂ : LinearMap.BilinForm K V
    b₂ : B₂.Nondegenerate
    v : V
    ⊢ Eq (B₂ ((B₁.symmCompOfNondegenerate B₂ b₂) v)) (B₁ v)
  -/
  rw [symmCompOfNondegenerate]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B₁ B₂ : LinearMap.BilinForm K V
    b₂ : B₂.Nondegenerate
    v : V
    ⊢ Eq (B₂ (((↑(B₂.toDual b₂).symm).comp B₁) v)) (B₁ v)
  -/
  simp only [coe_comp, LinearEquiv.coe_coe, Function.comp_apply, DFunLike.coe_fn_eq]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B₁ B₂ : LinearMap.BilinForm K V
    b₂ : B₂.Nondegenerate
    v : V
    ⊢ Eq (B₂ ((B₂.toDual b₂).symm (B₁ v))) (B₁ v)
  -/
  erw [LinearEquiv.apply_symm_apply (B₂.toDual b₂)]
  /-
    🎉 no goals
  -/


@[simp]
theorem symmCompOfNondegenerate_left_apply (B₁ : BilinForm K V) {B₂ : BilinForm K V}
    (b₂ : B₂.Nondegenerate) (v w : V) : B₂ (symmCompOfNondegenerate B₁ B₂ b₂ w) v = B₁ w v := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B₁ B₂ : LinearMap.BilinForm K V
    b₂ : B₂.Nondegenerate
    v w : V
    ⊢ Eq ((B₂ ((B₁.symmCompOfNondegenerate B₂ b₂) w)) v) ((B₁ w) v)
  -/
  conv_lhs => rw [comp_symmCompOfNondegenerate_apply]
  /-
    🎉 no goals
  -/


/-- Given the nondegenerate bilinear form `B` and the linear map `φ`,
`leftAdjointOfNondegenerate` provides the left adjoint of `φ` with respect to `B`.
The lemma proving this property is `BilinForm.isAdjointPairLeftAdjointOfNondegenerate`. -/
noncomputable def leftAdjointOfNondegenerate (B : BilinForm K V) (b : B.Nondegenerate)
    (φ : V →ₗ[K] V) : V →ₗ[K] V :=
  symmCompOfNondegenerate (B.compRight φ) B b


theorem isAdjointPairLeftAdjointOfNondegenerate (B : BilinForm K V) (b : B.Nondegenerate)
    (φ : V →ₗ[K] V) : IsAdjointPair B B (B.leftAdjointOfNondegenerate b φ) φ := fun x y =>
  (B.compRight φ).symmCompOfNondegenerate_left_apply b y x


/-- Given the nondegenerate bilinear form `B`, the linear map `φ` has a unique left adjoint given by
`BilinForm.leftAdjointOfNondegenerate`. -/
theorem isAdjointPair_iff_eq_of_nondegenerate (B : BilinForm K V) (b : B.Nondegenerate)
    (ψ φ : V →ₗ[K] V) : IsAdjointPair B B ψ φ ↔ ψ = B.leftAdjointOfNondegenerate b φ :=
  ⟨fun h =>
    B.isAdjointPair_unique_of_nondegenerate b φ ψ _ h
      (isAdjointPairLeftAdjointOfNondegenerate _ _ _),
    fun h => h.symm ▸ isAdjointPairLeftAdjointOfNondegenerate _ _ _⟩


