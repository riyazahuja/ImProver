/-- Map from ideals of `R` to submodules of `S` induced by `f`. -/
def coeSubmodule (I : Ideal R) : Submodule R S :=
  Submodule.map (Algebra.linearMap R S) I


theorem mem_coeSubmodule (I : Ideal R) {x : S} :
    x ∈ coeSubmodule S I ↔ ∃ y : R, y ∈ I ∧ algebraMap R S y = x :=
  Iff.rfl


theorem coeSubmodule_mono {I J : Ideal R} (h : I ≤ J) : coeSubmodule S I ≤ coeSubmodule S J :=
  Submodule.map_mono h


@[simp]
theorem coeSubmodule_bot : coeSubmodule S (⊥ : Ideal R) = ⊥ := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    ⊢ Eq (IsLocalization.coeSubmodule S Bot.bot) Bot.bot
  -/
  rw [coeSubmodule, Submodule.map_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeSubmodule_top : coeSubmodule S (⊤ : Ideal R) = 1 := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    ⊢ Eq (IsLocalization.coeSubmodule S Top.top) 1
  -/
  rw [coeSubmodule, Submodule.map_top, Submodule.one_eq_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeSubmodule_sup (I J : Ideal R) :
    coeSubmodule S (I ⊔ J) = coeSubmodule S I ⊔ coeSubmodule S J :=
  Submodule.map_sup _ _ _


@[simp]
theorem coeSubmodule_mul (I J : Ideal R) :
    coeSubmodule S (I * J) = coeSubmodule S I * coeSubmodule S J :=
  Submodule.map_mul _ _ (Algebra.ofId R S)


theorem coeSubmodule_fg (hS : Function.Injective (algebraMap R S)) (I : Ideal R) :
    Submodule.FG (coeSubmodule S I) ↔ Submodule.FG I :=
  ⟨Submodule.fg_of_fg_map_injective _ hS, Submodule.FG.map _⟩


@[simp]
theorem coeSubmodule_span (s : Set R) :
    coeSubmodule S (Ideal.span s) = Submodule.span R (algebraMap R S '' s) := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    s : Set R
    ⊢ Eq (IsLocalization.coeSubmodule S (Ideal.span s)) (Submodule.span R (Set.ima …
  -/
  rw [IsLocalization.coeSubmodule, Ideal.span, Submodule.map_span]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    s : Set R
    ⊢ Eq (Submodule.span R (Set.image (⇑(Algebra.linearMap R S)) s)) (Submodule.sp …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coeSubmodule_span_singleton (x : R) :
    coeSubmodule S (Ideal.span {x}) = Submodule.span R {(algebraMap R S) x} := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    x : R
    ⊢ Eq (IsLocalization.coeSubmodule S (Ideal.span (Singleton.singleton x))) (Sub …
  -/
  rw [coeSubmodule_span, Set.image_singleton]
  /-
    🎉 no goals
  -/


include M in
theorem isNoetherianRing (h : IsNoetherianRing R) : IsNoetherianRing S := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    h : IsNoetherianRing R
    ⊢ IsNoetherianRing S
  -/
  rw [isNoetherianRing_iff, isNoetherian_iff] at h ⊢
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    h : WellFounded fun x1 x2 => GT.gt x1 x2
    ⊢ WellFounded fun x1 x2 => GT.gt x1 x2
  -/
  exact OrderEmbedding.wellFounded (IsLocalization.orderEmbedding M S).dual h
  /-
    🎉 no goals
  -/


@[mono]
theorem coeSubmodule_le_coeSubmodule (h : M ≤ nonZeroDivisors R) {I J : Ideal R} :
    coeSubmodule S I ≤ coeSubmodule S J ↔ I ≤ J :=
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to specify the value of `f` here:
  Submodule.map_le_map_iff_of_injective (f := Algebra.linearMap R S) (IsLocalization.injective _ h)
    _ _


@[mono]
theorem coeSubmodule_strictMono (h : M ≤ nonZeroDivisors R) :
    StrictMono (coeSubmodule S : Ideal R → Submodule R S) :=
  strictMono_of_le_iff_le fun _ _ => (coeSubmodule_le_coeSubmodule h).symm


theorem coeSubmodule_injective (h : M ≤ nonZeroDivisors R) :
    Function.Injective (coeSubmodule S : Ideal R → Submodule R S) :=
  injective_of_le_imp_le _ fun hl => (coeSubmodule_le_coeSubmodule h).mp hl


theorem coeSubmodule_isPrincipal {I : Ideal R} (h : M ≤ nonZeroDivisors R) :
    (coeSubmodule S I).IsPrincipal ↔ I.IsPrincipal := by
  /-
    R : Type u_3
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_4
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    I : Ideal R
    h : LE.le M (nonZeroDivisors R)
    ⊢ Iff (IsLocalization.coeSubmodule S I).IsPrincipal (Submodule.IsPrincipal I)
  -/
  constructor <;> rintro ⟨⟨x, hx⟩⟩
    /-
      case mp.mk.intro
      R : Type u_3
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_4
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      h : LE.le M (nonZeroDivisors R)
      x : S
      hx : Eq (IsLocalization.coeSubmodule S I) (Submodule.span R (Singleton.singlet …
      ⊢ Submodule.IsPrincipal I
    -/
  · have x_mem : x ∈ coeSubmodule S I := hx.symm ▸ Submodule.mem_span_singleton_self x
    /-
      case mp.mk.intro
      R : Type u_3
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_4
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      h : LE.le M (nonZeroDivisors R)
      x : S
      hx : Eq (IsLocalization.coeSubmodule S I) (Submodule.span R (Singleton.singlet …
      x_mem : Membership.mem (IsLocalization.coeSubmodule S I) x
      ⊢ Submodule.IsPrincipal I
    -/
    obtain ⟨x, _, rfl⟩ := (mem_coeSubmodule _ _).mp x_mem
    /-
      case mp.mk.intro.intro.intro
      R : Type u_3
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_4
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      h : LE.le M (nonZeroDivisors R)
      x : R
      left✝ : Membership.mem I x
      hx : Eq (IsLocalization.coeSubmodule S I) (Submodule.span R (Singleton.singlet …
      x_mem : Membership.mem (IsLocalization.coeSubmodule S I) ((algebraMap R S) x)
      ⊢ Submodule.IsPrincipal I
    -/
    refine ⟨⟨x, coeSubmodule_injective S h ?_⟩⟩
    /-
      case mp.mk.intro.intro.intro
      R : Type u_3
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_4
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      h : LE.le M (nonZeroDivisors R)
      x : R
      left✝ : Membership.mem I x
      hx : Eq (IsLocalization.coeSubmodule S I) (Submodule.span R (Singleton.singlet …
      x_mem : Membership.mem (IsLocalization.coeSubmodule S I) ((algebraMap R S) x)
      ⊢ Eq (IsLocalization.coeSubmodule S I) (IsLocalization.coeSubmodule S (Submodu …
    -/
    rw [Ideal.submodule_span_eq, hx, coeSubmodule_span_singleton]
    /-
      🎉 no goals
    -/
    /-
      case mpr.mk.intro
      R : Type u_3
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_4
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      h : LE.le M (nonZeroDivisors R)
      x : R
      hx : Eq I (Submodule.span R (Singleton.singleton x))
      ⊢ (IsLocalization.coeSubmodule S I).IsPrincipal
    -/
  · refine ⟨⟨algebraMap R S x, ?_⟩⟩
    /-
      case mpr.mk.intro
      R : Type u_3
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_4
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      I : Ideal R
      h : LE.le M (nonZeroDivisors R)
      x : R
      hx : Eq I (Submodule.span R (Singleton.singleton x))
      ⊢ Eq (IsLocalization.coeSubmodule S I) (Submodule.span R (Singleton.singleton  …
    -/
    rw [hx, Ideal.submodule_span_eq, coeSubmodule_span_singleton]
    /-
      🎉 no goals
    -/


theorem mem_span_iff {N : Type*} [AddCommMonoid N] [Module R N] [Module S N] [IsScalarTower R S N]
    {x : N} {a : Set N} :
    x ∈ Submodule.span S a ↔ ∃ y ∈ Submodule.span R a, ∃ z : M, x = mk' S 1 z • y := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommSemiring S
    inst✝⁵ : Algebra R S
    inst✝⁴ : IsLocalization M S
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    x : N
    a : Set N
    ⊢ Iff (Membership.mem (Submodule.span S a) x) (Exists fun y => And (Membership …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      inst✝⁴ : IsLocalization M S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      x : N
      a : Set N
      ⊢ Membership.mem (Submodule.span S a) x → Exists fun y => And (Membership.mem  …
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      inst✝⁴ : IsLocalization M S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      x : N
      a : Set N
      h : Membership.mem (Submodule.span S a) x
      ⊢ Exists fun y => And (Membership.mem (Submodule.span R a) y) (Exists fun z => …
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ h
      /-
        case mp.refine_1
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x
        ⊢ ∀ (x : N), Membership.mem a x → Exists fun y => And (Membership.mem (Submodu …
      -/
    · rintro x hx
      /-
        case mp.refine_1
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x✝ : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x✝
        x : N
        hx : Membership.mem a x
        ⊢ Exists fun y => And (Membership.mem (Submodule.span R a) y) (Exists fun z => …
      -/
      exact ⟨x, Submodule.subset_span hx, 1, by rw [mk'_one, map_one, one_smul]⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_2
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x
        ⊢ Exists fun y => And (Membership.mem (Submodule.span R a) y) (Exists fun z => …
      -/
    · exact ⟨0, Submodule.zero_mem _, 1, by rw [mk'_one, map_one, one_smul]⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.refine_3
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x
        ⊢ ∀ (x y : N), Membership.mem (Submodule.span S a) x → Membership.mem (Submodu …
      -/
    · rintro _ _ _ _ ⟨y, hy, z, rfl⟩ ⟨y', hy', z', rfl⟩
      refine
        ⟨(z' : R) • y + (z : R) • y',
          Submodule.add_mem _ (Submodule.smul_mem _ _ hy) (Submodule.smul_mem _ _ hy'), z * z', ?_⟩
      rw [smul_add, ← IsScalarTower.algebraMap_smul S (z : R), ←
        IsScalarTower.algebraMap_smul S (z' : R), smul_smul, smul_smul]
      /-
        case mp.refine_3.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x
        y : N
        hy : Membership.mem (Submodule.span R a) y
        z : Subtype fun x => Membership.mem M x
        hx✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
        y' : N
        hy' : Membership.mem (Submodule.span R a) y'
        z' : Subtype fun x => Membership.mem M x
        hy✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul (IsLocalization.mk' S 1 z) y) (HSMul.hSMul (IsLoc …
      -/
      congr 1
        /-
          case mp.refine_3.intro.intro.intro.intro.intro.intro.e_a
          R : Type u_1
          inst✝⁷ : CommSemiring R
          M : Submonoid R
          S : Type u_2
          inst✝⁶ : CommSemiring S
          inst✝⁵ : Algebra R S
          inst✝⁴ : IsLocalization M S
          N : Type u_3
          inst✝³ : AddCommMonoid N
          inst✝² : Module R N
          inst✝¹ : Module S N
          inst✝ : IsScalarTower R S N
          x : N
          a : Set N
          h : Membership.mem (Submodule.span S a) x
          y : N
          hy : Membership.mem (Submodule.span R a) y
          z : Subtype fun x => Membership.mem M x
          hx✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
          y' : N
          hy' : Membership.mem (Submodule.span R a) y'
          z' : Subtype fun x => Membership.mem M x
          hy✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
          ⊢ Eq (HSMul.hSMul (IsLocalization.mk' S 1 z) y) (HSMul.hSMul (HMul.hMul (IsLoc …
        -/
      · rw [← mul_one (1 : R), mk'_mul, mul_assoc, mk'_spec, map_one, mul_one, mul_one]
        /-
          🎉 no goals
        -/
        /-
          case mp.refine_3.intro.intro.intro.intro.intro.intro.e_a
          R : Type u_1
          inst✝⁷ : CommSemiring R
          M : Submonoid R
          S : Type u_2
          inst✝⁶ : CommSemiring S
          inst✝⁵ : Algebra R S
          inst✝⁴ : IsLocalization M S
          N : Type u_3
          inst✝³ : AddCommMonoid N
          inst✝² : Module R N
          inst✝¹ : Module S N
          inst✝ : IsScalarTower R S N
          x : N
          a : Set N
          h : Membership.mem (Submodule.span S a) x
          y : N
          hy : Membership.mem (Submodule.span R a) y
          z : Subtype fun x => Membership.mem M x
          hx✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
          y' : N
          hy' : Membership.mem (Submodule.span R a) y'
          z' : Subtype fun x => Membership.mem M x
          hy✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
          ⊢ Eq (HSMul.hSMul (IsLocalization.mk' S 1 z') y') (HSMul.hSMul (HMul.hMul (IsL …
        -/
      · rw [← mul_one (1 : R), mk'_mul, mul_right_comm, mk'_spec, map_one, mul_one, one_mul]
        /-
          🎉 no goals
        -/
      /-
        case mp.refine_4
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x
        ⊢ ∀ (a_1 : S) (x : N), Membership.mem (Submodule.span S a) x → (Exists fun y = …
      -/
    · rintro a _ _ ⟨y, hy, z, rfl⟩
      /-
        case mp.refine_4.intro.intro.intro
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a✝ : Set N
        h : Membership.mem (Submodule.span S a✝) x
        a : S
        y : N
        hy : Membership.mem (Submodule.span R a✝) y
        z : Subtype fun x => Membership.mem M x
        hx✝ : Membership.mem (Submodule.span S a✝) (HSMul.hSMul (IsLocalization.mk' S  …
        ⊢ Exists fun y_1 => And (Membership.mem (Submodule.span R a✝) y_1) (Exists fun …
      -/
      obtain ⟨y', z', rfl⟩ := mk'_surjective M a
      /-
        case mp.refine_4.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        inst✝⁴ : IsLocalization M S
        N : Type u_3
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module S N
        inst✝ : IsScalarTower R S N
        x : N
        a : Set N
        h : Membership.mem (Submodule.span S a) x
        y : N
        hy : Membership.mem (Submodule.span R a) y
        z : Subtype fun x => Membership.mem M x
        hx✝ : Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 …
        y' : R
        z' : Subtype fun x => Membership.mem M x
        ⊢ Exists fun y_1 => And (Membership.mem (Submodule.span R a) y_1) (Exists fun  …
      -/
      refine ⟨y' • y, Submodule.smul_mem _ _ hy, z' * z, ?_⟩
      rw [← IsScalarTower.algebraMap_smul S y', smul_smul, ← mk'_mul, smul_smul,
        mul_comm (mk' S _ _), mul_mk'_eq_mk'_of_mul]
    /-
      case mpr
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      inst✝⁴ : IsLocalization M S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      x : N
      a : Set N
      ⊢ (Exists fun y => And (Membership.mem (Submodule.span R a) y) (Exists fun z = …
    -/
  · rintro ⟨y, hy, z, rfl⟩
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      inst✝⁴ : IsLocalization M S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      a : Set N
      y : N
      hy : Membership.mem (Submodule.span R a) y
      z : Subtype fun x => Membership.mem M x
      ⊢ Membership.mem (Submodule.span S a) (HSMul.hSMul (IsLocalization.mk' S 1 z) y)
    -/
    exact Submodule.smul_mem _ _ (Submodule.span_subset_span R S _ hy)
    /-
      🎉 no goals
    -/


theorem mem_span_map {x : S} {a : Set R} :
    x ∈ Ideal.span (algebraMap R S '' a) ↔ ∃ y ∈ Ideal.span a, ∃ z : M, x = mk' S y z := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : S
    a : Set R
    ⊢ Iff (Membership.mem (Ideal.span (Set.image (⇑(algebraMap R S)) a)) x) (Exist …
  -/
  refine (mem_span_iff M).trans ?_
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : S
    a : Set R
    ⊢ Iff (Exists fun y => And (Membership.mem (Submodule.span R (Set.image (⇑(alg …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      ⊢ (Exists fun y => And (Membership.mem (Submodule.span R (Set.image (⇑(algebra …
    -/
  · rw [← coeSubmodule_span]
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      ⊢ (Exists fun y => And (Membership.mem (IsLocalization.coeSubmodule S (Ideal.s …
    -/
    rintro ⟨_, ⟨y, hy, rfl⟩, z, hz⟩
    /-
      case mp.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      y : R
      hy : Membership.mem (↑(Ideal.span a)) y
      z : Subtype fun x => Membership.mem M x
      hz : Eq x (HSMul.hSMul (IsLocalization.mk' S 1 z) ((Algebra.linearMap R S) y))
      ⊢ Exists fun y => And (Membership.mem (Ideal.span a) y) (Exists fun z => Eq x  …
    -/
    refine ⟨y, hy, z, ?_⟩
    /-
      case mp.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      y : R
      hy : Membership.mem (↑(Ideal.span a)) y
      z : Subtype fun x => Membership.mem M x
      hz : Eq x (HSMul.hSMul (IsLocalization.mk' S 1 z) ((Algebra.linearMap R S) y))
      ⊢ Eq x (IsLocalization.mk' S y z)
    -/
    rw [hz, Algebra.linearMap_apply, smul_eq_mul, mul_comm, mul_mk'_eq_mk'_of_mul, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      ⊢ (Exists fun y => And (Membership.mem (Ideal.span a) y) (Exists fun z => Eq x …
    -/
  · rintro ⟨y, hy, z, hz⟩
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      y : R
      hy : Membership.mem (Ideal.span a) y
      z : Subtype fun x => Membership.mem M x
      hz : Eq x (IsLocalization.mk' S y z)
      ⊢ Exists fun y => And (Membership.mem (Submodule.span R (Set.image (⇑(algebraM …
    -/
    refine ⟨algebraMap R S y, Submodule.map_mem_span_algebraMap_image _ _ hy, z, ?_⟩
    /-
      case mpr.intro.intro.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : S
      a : Set R
      y : R
      hy : Membership.mem (Ideal.span a) y
      z : Subtype fun x => Membership.mem M x
      hz : Eq x (IsLocalization.mk' S y z)
      ⊢ Eq x (HSMul.hSMul (IsLocalization.mk' S 1 z) ((algebraMap R S) y))
    -/
    rw [hz, smul_eq_mul, mul_comm, mul_mk'_eq_mk'_of_mul, mul_one]
    /-
      🎉 no goals
    -/


@[simp, mono]
theorem coeSubmodule_le_coeSubmodule {I J : Ideal R} :
    coeSubmodule K I ≤ coeSubmodule K J ↔ I ≤ J :=
  IsLocalization.coeSubmodule_le_coeSubmodule le_rfl


@[mono]
theorem coeSubmodule_strictMono : StrictMono (coeSubmodule K : Ideal R → Submodule R K) :=
  strictMono_of_le_iff_le fun _ _ => coeSubmodule_le_coeSubmodule.symm


theorem coeSubmodule_injective : Function.Injective (coeSubmodule K : Ideal R → Submodule R K) :=
  injective_of_le_imp_le _ fun hl => coeSubmodule_le_coeSubmodule.mp hl


@[simp]
theorem coeSubmodule_isPrincipal {I : Ideal R} : (coeSubmodule K I).IsPrincipal ↔ I.IsPrincipal :=
  IsLocalization.coeSubmodule_isPrincipal _ le_rfl


