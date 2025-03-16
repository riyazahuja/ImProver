lemma isNilpotent_iff_isNilpotent_fst {x : TrivSqZeroExt R M} :
    IsNilpotent x ↔ IsNilpotent x.fst := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : SMulCommClass R (MulOpposite R) M
    x : TrivSqZeroExt R M
    ⊢ Iff (IsNilpotent x) (IsNilpotent x.fst)
  -/
  constructor <;> rintro ⟨n, hn⟩
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      x : TrivSqZeroExt R M
      n : Nat
      hn : Eq (HPow.hPow x n) 0
      ⊢ IsNilpotent x.fst
    -/
  · refine ⟨n, ?_⟩
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      x : TrivSqZeroExt R M
      n : Nat
      hn : Eq (HPow.hPow x n) 0
      ⊢ Eq (HPow.hPow x.fst n) 0
    -/
    rw [← fst_pow, hn, fst_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      x : TrivSqZeroExt R M
      n : Nat
      hn : Eq (HPow.hPow x.fst n) 0
      ⊢ IsNilpotent x
    -/
  · refine ⟨n * 2, ?_⟩
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      x : TrivSqZeroExt R M
      n : Nat
      hn : Eq (HPow.hPow x.fst n) 0
      ⊢ Eq (HPow.hPow x (HMul.hMul n 2)) 0
    -/
    rw [pow_mul]
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module (MulOpposite R) M
      inst✝ : SMulCommClass R (MulOpposite R) M
      x : TrivSqZeroExt R M
      n : Nat
      hn : Eq (HPow.hPow x.fst n) 0
      ⊢ Eq (HPow.hPow (HPow.hPow x n) 2) 0
    -/
    ext
      /-
        case mpr.intro.h1
        R : Type u_1
        M : Type u_2
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : Module (MulOpposite R) M
        inst✝ : SMulCommClass R (MulOpposite R) M
        x : TrivSqZeroExt R M
        n : Nat
        hn : Eq (HPow.hPow x.fst n) 0
        ⊢ Eq (HPow.hPow (HPow.hPow x n) 2).fst (TrivSqZeroExt.fst 0)
      -/
    · rw [fst_pow, fst_pow, hn, zero_pow two_ne_zero, fst_zero]
      /-
        🎉 no goals
      -/
    · rw [pow_two, snd_mul, fst_pow, hn, MulOpposite.op_zero, zero_smul, zero_smul, zero_add,
        snd_zero]


@[simp]
lemma isNilpotent_inl_iff (r : R) : IsNilpotent (.inl r : TrivSqZeroExt R M) ↔ IsNilpotent r := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : SMulCommClass R (MulOpposite R) M
    r : R
    ⊢ Iff (IsNilpotent (TrivSqZeroExt.inl r)) (IsNilpotent r)
  -/
  rw [isNilpotent_iff_isNilpotent_fst, fst_inl]
  /-
    🎉 no goals
  -/


@[simp]
lemma isNilpotent_inr (x : M) : IsNilpotent (.inr x : TrivSqZeroExt R M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : SMulCommClass R (MulOpposite R) M
    x : M
    ⊢ IsNilpotent (TrivSqZeroExt.inr x)
  -/
  refine ⟨2, by simp [pow_two]⟩
  /-
    🎉 no goals
  -/


lemma isUnit_or_isNilpotent_of_isMaximal_isNilpotent [CommSemiring R] [AddCommGroup M]
    [Module R M] [Module Rᵐᵒᵖ M] [IsCentralScalar R M]
    (h : ∀ I : Ideal R, I.IsMaximal → IsNilpotent I)
    (a : TrivSqZeroExt R M) :
    IsUnit a ∨ IsNilpotent a := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : IsCentralScalar R M
    h : ∀ (I : Ideal R), I.IsMaximal → IsNilpotent I
    a : TrivSqZeroExt R M
    ⊢ Or (IsUnit a) (IsNilpotent a)
  -/
  rw [isUnit_iff_isUnit_fst, isNilpotent_iff_isNilpotent_fst]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : IsCentralScalar R M
    h : ∀ (I : Ideal R), I.IsMaximal → IsNilpotent I
    a : TrivSqZeroExt R M
    ⊢ Or (IsUnit a.fst) (IsNilpotent a.fst)
  -/
  refine (em _).imp_right fun ha ↦ ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : IsCentralScalar R M
    h : ∀ (I : Ideal R), I.IsMaximal → IsNilpotent I
    a : TrivSqZeroExt R M
    ha : Not (IsUnit a.fst)
    ⊢ IsNilpotent a.fst
  -/
  obtain ⟨I, hI, haI⟩ := exists_max_ideal_of_mem_nonunits (mem_nonunits_iff.mpr ha)
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : IsCentralScalar R M
    h : ∀ (I : Ideal R), I.IsMaximal → IsNilpotent I
    a : TrivSqZeroExt R M
    ha : Not (IsUnit a.fst)
    I : Ideal R
    hI : I.IsMaximal
    haI : Membership.mem I a.fst
    ⊢ IsNilpotent a.fst
  -/
  refine (h _ hI).imp fun n hn ↦ ?_
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : IsCentralScalar R M
    h : ∀ (I : Ideal R), I.IsMaximal → IsNilpotent I
    a : TrivSqZeroExt R M
    ha : Not (IsUnit a.fst)
    I : Ideal R
    hI : I.IsMaximal
    haI : Membership.mem I a.fst
    n : Nat
    hn : Eq (HPow.hPow I n) 0
    ⊢ Eq (HPow.hPow a.fst n) 0
  -/
  exact hn.le (Ideal.pow_mem_pow haI _)
  /-
    🎉 no goals
  -/


lemma isUnit_or_isNilpotent [DivisionSemiring R] [AddCommGroup M]
    [Module R M] [Module Rᵐᵒᵖ M] [SMulCommClass R Rᵐᵒᵖ M]
    (a : TrivSqZeroExt R M) :
    IsUnit a ∨ IsNilpotent a := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : DivisionSemiring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module (MulOpposite R) M
    inst✝ : SMulCommClass R (MulOpposite R) M
    a : TrivSqZeroExt R M
    ⊢ Or (IsUnit a) (IsNilpotent a)
  -/
  simp [isUnit_iff_isUnit_fst, isNilpotent_iff_isNilpotent_fst, em']
  /-
    🎉 no goals
  -/


lemma fst_eq_zero_iff_eps_dvd [Semiring R] {x : R[ε]} :
    x.fst = 0 ↔ ε ∣ x := by
  simp_rw [dvd_def, TrivSqZeroExt.ext_iff, TrivSqZeroExt.fst_mul, TrivSqZeroExt.snd_mul,
    fst_eps, snd_eps, zero_mul, zero_smul, zero_add, MulOpposite.smul_eq_mul_unop,
    MulOpposite.unop_op, one_mul, exists_and_left, iff_self_and]
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : DualNumber R
    ⊢ Eq (TrivSqZeroExt.fst x) 0 → Exists fun x_1 => Eq (TrivSqZeroExt.snd x) (Tri …
  -/
  intro
  /-
    R : Type u_1
    inst✝ : Semiring R
    x : DualNumber R
    a✝ : Eq (TrivSqZeroExt.fst x) 0
    ⊢ Exists fun x_1 => Eq (TrivSqZeroExt.snd x) (TrivSqZeroExt.fst x_1)
  -/
  exact ⟨.inl x.snd, rfl⟩
  /-
    🎉 no goals
  -/


lemma isNilpotent_eps [Semiring R] :
    IsNilpotent (ε : R[ε]) :=
  TrivSqZeroExt.isNilpotent_inr 1


lemma isNilpotent_iff_eps_dvd [DivisionSemiring R] {x : R[ε]} :
    IsNilpotent x ↔ ε ∣ x := by
  /-
    R : Type u_1
    inst✝ : DivisionSemiring R
    x : DualNumber R
    ⊢ Iff (IsNilpotent x) (Dvd.dvd DualNumber.eps x)
  -/
  simp only [isNilpotent_iff_isNilpotent_fst, isNilpotent_iff_eq_zero, fst_eq_zero_iff_eps_dvd]
  /-
    🎉 no goals
  -/


instance [DivisionRing K] : IsLocalRing K[ε] where
  isUnit_or_isUnit_of_add_one {a b} h := by
    /-
      R : Type u_1
      K : Type u_2
      inst✝ : DivisionRing K
      a b : DualNumber K
      h : Eq (HAdd.hAdd a b) 1
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
    rw [add_comm, ← eq_sub_iff_add_eq] at h
    /-
      R : Type u_1
      K : Type u_2
      inst✝ : DivisionRing K
      a b : DualNumber K
      h : Eq b (HSub.hSub 1 a)
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
    rcases eq_or_ne (fst a) 0 with ha|ha <;>
    /-
      case inl
      R : Type u_1
      K : Type u_2
      inst✝ : DivisionRing K
      a b : DualNumber K
      h : Eq b (HSub.hSub 1 a)
      ha : Eq (TrivSqZeroExt.fst a) 0
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
    /-
      🎉 no goals
    -/
    simp [isUnit_iff_isUnit_fst, h, ha]
    /-
      🎉 no goals
    -/


lemma ideal_trichotomy [DivisionRing K] (I : Ideal K[ε]) :
    I = ⊥ ∨ I = .span {ε} ∨ I = ⊤ := by
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    I : Ideal (DualNumber K)
    ⊢ Or (Eq I Bot.bot) (Or (Eq I (Ideal.span (Singleton.singleton DualNumber.eps) …
  -/
  refine (eq_or_ne I ⊥).imp_right fun hb ↦ ?_
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    I : Ideal (DualNumber K)
    hb : Ne I Bot.bot
    ⊢ Or (Eq I (Ideal.span (Singleton.singleton DualNumber.eps))) (Eq I Top.top)
  -/
  refine (eq_or_ne I ⊤).symm.imp_left fun ht ↦ ?_
  have hd : ∀ x ∈ I, ε ∣ x := by
    intro x hxI
    rcases isUnit_or_isNilpotent x with hx|hx
    · exact absurd (Ideal.eq_top_of_isUnit_mem _ hxI hx) ht
    · rwa [← isNilpotent_iff_eps_dvd]
  have hd' : ∀ x ∈ I, x ≠ 0 → ∃ r, ε = r * x := by
    intro x hxI hx0
    obtain ⟨r, rfl⟩ := hd _ hxI
    have : ε * r = (fst r) • ε := by ext <;> simp
    rw [this] at hxI hx0 ⊢
    have hr : fst r ≠ 0 := by
      contrapose! hx0
      simp [hx0]
    refine ⟨r⁻¹, ?_⟩
    simp [TrivSqZeroExt.ext_iff, inv_mul_cancel₀ hr]
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    I : Ideal (DualNumber K)
    hb : Ne I Bot.bot
    ht : Ne I Top.top
    hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
    hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
    ⊢ Eq I (Ideal.span (Singleton.singleton DualNumber.eps))
  -/
  refine le_antisymm ?_ ?_ <;> intro x <;>
    /-
      case refine_1
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      x : DualNumber K
      ⊢ Membership.mem I x → Membership.mem (Ideal.span (Singleton.singleton DualNum …
    -/
    simp_rw [Ideal.mem_span_singleton', (commute_eps_right _).eq, eq_comm, ← dvd_def]
    /-
      case refine_1
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      x : DualNumber K
      ⊢ Membership.mem I x → Dvd.dvd DualNumber.eps x
    -/
  · intro hx
    /-
      case refine_1
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      x : DualNumber K
      hx : Membership.mem I x
      ⊢ Dvd.dvd DualNumber.eps x
    -/
    simp_rw [hd _ hx]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      x : DualNumber K
      ⊢ Dvd.dvd DualNumber.eps x → Membership.mem I x
    -/
  · intro hx
    /-
      case refine_2
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      x : DualNumber K
      hx : Dvd.dvd DualNumber.eps x
      ⊢ Membership.mem I x
    -/
    obtain ⟨p, rfl⟩ := hx
    /-
      case refine_2.intro
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      p : DualNumber K
      ⊢ Membership.mem I (HMul.hMul DualNumber.eps p)
    -/
    obtain ⟨y, hyI, hy0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hb
    /-
      case refine_2.intro.intro.intro
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      p y : DualNumber K
      hyI : Membership.mem I y
      hy0 : Ne y 0
      ⊢ Membership.mem I (HMul.hMul DualNumber.eps p)
    -/
    obtain ⟨r, hr⟩ := hd' _ hyI hy0
    /-
      case refine_2.intro.intro.intro.intro
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      p y : DualNumber K
      hyI : Membership.mem I y
      hy0 : Ne y 0
      r : DualNumber K
      hr : Eq DualNumber.eps (HMul.hMul r y)
      ⊢ Membership.mem I (HMul.hMul DualNumber.eps p)
    -/
    rw [(commute_eps_left _).eq, hr, ← mul_assoc]
    /-
      case refine_2.intro.intro.intro.intro
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hb : Ne I Bot.bot
      ht : Ne I Top.top
      hd : ∀ (x : DualNumber K), Membership.mem I x → Dvd.dvd DualNumber.eps x
      hd' : ∀ (x : DualNumber K), Membership.mem I x → Ne x 0 → Exists fun r => Eq D …
      p y : DualNumber K
      hyI : Membership.mem I y
      hy0 : Ne y 0
      r : DualNumber K
      hr : Eq DualNumber.eps (HMul.hMul r y)
      ⊢ Membership.mem I (HMul.hMul (HMul.hMul p r) y)
    -/
    exact Ideal.mul_mem_left _ _ hyI
    /-
      🎉 no goals
    -/


lemma isMaximal_span_singleton_eps [DivisionRing K] :
    (Ideal.span {ε} : Ideal K[ε]).IsMaximal := by
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    ⊢ (Ideal.span (Singleton.singleton DualNumber.eps)).IsMaximal
  -/
  refine ⟨?_, fun I hI ↦ ?_⟩
    /-
      case refine_1
      K : Type u_2
      inst✝ : DivisionRing K
      ⊢ Ne (Ideal.span (Singleton.singleton DualNumber.eps)) Top.top
    -/
  · simp [ne_eq, Ideal.eq_top_iff_one, Ideal.mem_span_singleton', TrivSqZeroExt.ext_iff]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      hI : LT.lt (Ideal.span (Singleton.singleton DualNumber.eps)) I
      ⊢ Eq I Top.top
    -/
  · rcases ideal_trichotomy I with rfl|rfl|rfl <;>
    /-
      case refine_2.inl
      K : Type u_2
      inst✝ : DivisionRing K
      hI : LT.lt (Ideal.span (Singleton.singleton DualNumber.eps)) Bot.bot
      ⊢ Eq Bot.bot Top.top
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    first | simp at hI | simp
    /-
      🎉 no goals
    -/


lemma maximalIdeal_eq_span_singleton_eps [Field K] :
    IsLocalRing.maximalIdeal K[ε] = Ideal.span {ε} :=
  (IsLocalRing.eq_maximalIdeal isMaximal_span_singleton_eps).symm


instance [DivisionRing K] : IsPrincipalIdealRing K[ε] where
  principal I := by
    /-
      R : Type u_1
      K : Type u_2
      inst✝ : DivisionRing K
      I : Ideal (DualNumber K)
      ⊢ Submodule.IsPrincipal I
    -/
    rcases ideal_trichotomy I with rfl|rfl|rfl
      /-
        case inl
        R : Type u_1
        K : Type u_2
        inst✝ : DivisionRing K
        ⊢ Submodule.IsPrincipal Bot.bot
      -/
    · exact bot_isPrincipal
      /-
        🎉 no goals
      -/
      /-
        case inr.inl
        R : Type u_1
        K : Type u_2
        inst✝ : DivisionRing K
        ⊢ Submodule.IsPrincipal (Ideal.span (Singleton.singleton DualNumber.eps))
      -/
    · exact ⟨_, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        R : Type u_1
        K : Type u_2
        inst✝ : DivisionRing K
        ⊢ Submodule.IsPrincipal Top.top
      -/
    · exact top_isPrincipal
      /-
        🎉 no goals
      -/


lemma exists_mul_left_or_mul_right [DivisionRing K] (a b : K[ε]) :
    ∃ c, a * c = b ∨ b * c = a := by
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    a b : DualNumber K
    ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
  -/
  rcases isUnit_or_isNilpotent a with ha|ha
    /-
      case inl
      K : Type u_2
      inst✝ : DivisionRing K
      a b : DualNumber K
      ha : IsUnit a
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
  · lift a to K[ε]ˣ using ha
    /-
      case inl.intro
      K : Type u_2
      inst✝ : DivisionRing K
      b : DualNumber K
      a : Units (DualNumber K)
      ⊢ Exists fun c => Or (Eq (HMul.hMul (↑a) c) b) (Eq (HMul.hMul b c) ↑a)
    -/
    exact ⟨a⁻¹ * b, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    K : Type u_2
    inst✝ : DivisionRing K
    a b : DualNumber K
    ha : IsNilpotent a
    ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
  -/
  rcases isUnit_or_isNilpotent b with hb|hb
    /-
      case inr.inl
      K : Type u_2
      inst✝ : DivisionRing K
      a b : DualNumber K
      ha : IsNilpotent a
      hb : IsUnit b
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
    -/
  · lift b to K[ε]ˣ using hb
    /-
      case inr.inl.intro
      K : Type u_2
      inst✝ : DivisionRing K
      a : DualNumber K
      ha : IsNilpotent a
      b : Units (DualNumber K)
      ⊢ Exists fun c => Or (Eq (HMul.hMul a c) ↑b) (Eq (HMul.hMul (↑b) c) a)
    -/
    exact ⟨b⁻¹ * a, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    K : Type u_2
    inst✝ : DivisionRing K
    a b : DualNumber K
    ha : IsNilpotent a
    hb : IsNilpotent b
    ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
  -/
  rw [isNilpotent_iff_eps_dvd] at ha hb
  /-
    case inr.inr
    K : Type u_2
    inst✝ : DivisionRing K
    a b : DualNumber K
    ha : Dvd.dvd DualNumber.eps a
    hb : Dvd.dvd DualNumber.eps b
    ⊢ Exists fun c => Or (Eq (HMul.hMul a c) b) (Eq (HMul.hMul b c) a)
  -/
  obtain ⟨x, rfl⟩ := ha
  /-
    case inr.inr.intro
    K : Type u_2
    inst✝ : DivisionRing K
    b : DualNumber K
    hb : Dvd.dvd DualNumber.eps b
    x : DualNumber K
    ⊢ Exists fun c => Or (Eq (HMul.hMul (HMul.hMul DualNumber.eps x) c) b) (Eq (HM …
  -/
  obtain ⟨y, rfl⟩ := hb
  suffices ∃ c, fst x * fst c = fst y ∨ fst y * fst c = fst x by
    simpa [TrivSqZeroExt.ext_iff] using this
  /-
    case inr.inr.intro.intro
    K : Type u_2
    inst✝ : DivisionRing K
    x y : DualNumber K
    ⊢ Exists fun c => Or (Eq (HMul.hMul (TrivSqZeroExt.fst x) c.fst) (TrivSqZeroEx …
  -/
  rcases eq_or_ne (fst x) 0 with hx|hx
    /-
      case inr.inr.intro.intro.inl
      K : Type u_2
      inst✝ : DivisionRing K
      x y : DualNumber K
      hx : Eq (TrivSqZeroExt.fst x) 0
      ⊢ Exists fun c => Or (Eq (HMul.hMul (TrivSqZeroExt.fst x) c.fst) (TrivSqZeroEx …
    -/
  · refine ⟨ε, Or.inr ?_⟩
    /-
      case inr.inr.intro.intro.inl
      K : Type u_2
      inst✝ : DivisionRing K
      x y : DualNumber K
      hx : Eq (TrivSqZeroExt.fst x) 0
      ⊢ Eq (HMul.hMul (TrivSqZeroExt.fst y) (TrivSqZeroExt.fst DualNumber.eps)) (Tri …
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr.intro.intro.inr
    K : Type u_2
    inst✝ : DivisionRing K
    x y : DualNumber K
    hx : Ne (TrivSqZeroExt.fst x) 0
    ⊢ Exists fun c => Or (Eq (HMul.hMul (TrivSqZeroExt.fst x) c.fst) (TrivSqZeroEx …
  -/
  refine ⟨inl ((fst x)⁻¹ * fst y), ?_⟩
  /-
    case inr.inr.intro.intro.inr
    K : Type u_2
    inst✝ : DivisionRing K
    x y : DualNumber K
    hx : Ne (TrivSqZeroExt.fst x) 0
    ⊢ Or (Eq (HMul.hMul (TrivSqZeroExt.fst x) (TrivSqZeroExt.inl (HMul.hMul (Inv.i …
  -/
  simp [← inl_mul, ← mul_assoc, mul_inv_cancel₀ hx]
  /-
    🎉 no goals
  -/


