/-- Let `N` be a localization of an `R`-module `M` at `p`.
This is the localization of an `R`-submodule of `M` viewed as an `R`-submodule of `N`. -/
def localized₀ : Submodule R N where
  carrier := { x | ∃ m ∈ M', ∃ s : p, IsLocalizedModule.mk' f m s = x }
  add_mem' := fun {x y} ⟨m, hm, s, hx⟩ ⟨n, hn, t, hy⟩ ↦ ⟨t • m + s • n, add_mem (M'.smul_mem t hm)
                                  /-
                                    R : Type u_1
                                    S : Type u_2
                                    M : Type u_3
                                    N : Type u_4
                                    inst✝¹⁰ : CommSemiring R
                                    inst✝⁹ : CommSemiring S
                                    inst✝⁸ : AddCommMonoid M
                                    inst✝⁷ : AddCommMonoid N
                                    inst✝⁶ : Module R M
                                    inst✝⁵ : Module R N
                                    inst✝⁴ : Algebra R S
                                    inst✝³ : Module S N
                                    inst✝² : IsScalarTower R S N
                                    p : Submonoid R
                                    inst✝¹ : IsLocalization p S
                                    f : LinearMap (RingHom.id R) M N
                                    inst✝ : IsLocalizedModule p f
                                    M' : Submodule R M
                                    x y : N
                                    x✝¹ : Membership.mem (setOf fun x => Exists fun m => And (Membership.mem M' m) …
                                    x✝ : Membership.mem (setOf fun x => Exists fun m => And (Membership.mem M' m)  …
                                    m : M
                                    hm : Membership.mem M' m
                                    s : Subtype fun x => Membership.mem p x
                                    hx : Eq (IsLocalizedModule.mk' f m s) x
                                    n : M
                                    hn : Membership.mem M' n
                                    t : Subtype fun x => Membership.mem p x
                                    hy : Eq (IsLocalizedModule.mk' f n t) y
                                    ⊢ Eq (IsLocalizedModule.mk' f (HAdd.hAdd (HSMul.hSMul t m) (HSMul.hSMul s n))  …
                                  -/
    (M'.smul_mem s hn), s * t, by rw [← hx, ← hy, IsLocalizedModule.mk'_add_mk']⟩
                                  /-
                                    🎉 no goals
                                  -/
                                     /-
                                       R : Type u_1
                                       S : Type u_2
                                       M : Type u_3
                                       N : Type u_4
                                       inst✝¹⁰ : CommSemiring R
                                       inst✝⁹ : CommSemiring S
                                       inst✝⁸ : AddCommMonoid M
                                       inst✝⁷ : AddCommMonoid N
                                       inst✝⁶ : Module R M
                                       inst✝⁵ : Module R N
                                       inst✝⁴ : Algebra R S
                                       inst✝³ : Module S N
                                       inst✝² : IsScalarTower R S N
                                       p : Submonoid R
                                       inst✝¹ : IsLocalization p S
                                       f : LinearMap (RingHom.id R) M N
                                       inst✝ : IsLocalizedModule p f
                                       M' : Submodule R M
                                       ⊢ Eq (IsLocalizedModule.mk' f 0 1) 0
                                     -/
  zero_mem' := ⟨0, zero_mem _, 1, by simp⟩
                                     /-
                                       🎉 no goals
                                     -/
  smul_mem' r x := by
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      r : R
      x : N
      ⊢ Membership.mem { carrier := setOf fun x => Exists fun m => And (Membership.m …
    -/
    rintro ⟨m, hm, s, hx⟩
    /-
      case intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      r : R
      x : N
      m : M
      hm : Membership.mem M' m
      s : Subtype fun x => Membership.mem p x
      hx : Eq (IsLocalizedModule.mk' f m s) x
      ⊢ Membership.mem { carrier := setOf fun x => Exists fun m => And (Membership.m …
    -/
    exact ⟨r • m, smul_mem M' _ hm, s, by rw [IsLocalizedModule.mk'_smul, hx]⟩
    /-
      🎉 no goals
    -/


/-- Let `S` be the localization of `R` at `p` and `N` be a localization of `M` at `p`.
This is the localization of an `R`-submodule of `M` viewed as an `S`-submodule of `N`. -/
def localized' : Submodule S N where
  __ := localized₀ p f M'
  smul_mem' := fun r x ⟨m, hm, s, hx⟩ ↦ by
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      r : S
      x : N
      x✝ : Membership.mem __spread✝⁻⁰.carrier x
      m : M
      hm : Membership.mem M' m
      s : Subtype fun x => Membership.mem p x
      hx : Eq (IsLocalizedModule.mk' f m s) x
      ⊢ Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul r x)
    -/
    have ⟨y, t, hyt⟩ := IsLocalization.mk'_surjective p r
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      r : S
      x : N
      x✝ : Membership.mem __spread✝⁻⁰.carrier x
      m : M
      hm : Membership.mem M' m
      s : Subtype fun x => Membership.mem p x
      hx : Eq (IsLocalizedModule.mk' f m s) x
      y : R
      t : Subtype fun x => Membership.mem p x
      hyt : Eq (IsLocalization.mk' S y t) r
      ⊢ Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul r x)
    -/
    exact ⟨y • m, M'.smul_mem y hm, t * s, by simp [← hyt, ← hx, IsLocalizedModule.mk'_smul_mk']⟩
    /-
      🎉 no goals
    -/


lemma mem_localized₀ (x : N) :
    x ∈ localized₀ p f M' ↔ ∃ m ∈ M', ∃ s : p, IsLocalizedModule.mk' f m s = x :=
  Iff.rfl


lemma mem_localized' (x : N) :
    x ∈ localized' S p f M' ↔ ∃ m ∈ M', ∃ s : p, IsLocalizedModule.mk' f m s = x :=
  Iff.rfl


/-- `localized₀` is the same as `localized'` considered as a submodule over the base ring. -/
lemma restrictScalars_localized' :
    (localized' S p f M').restrictScalars R = localized₀ p f M' :=
  rfl


/-- The localization of an `R`-submodule of `M` at `p` viewed as an `Rₚ`-submodule of `Mₚ`. -/
abbrev localized : Submodule (Localization p) (LocalizedModule p M) :=
  M'.localized' (Localization p) p (LocalizedModule.mkLinearMap p M)


@[simp]
lemma localized₀_bot : (⊥ : Submodule R M).localized₀ p f = ⊥ := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    ⊢ Eq (Submodule.localized₀ p f Bot.bot) Bot.bot
  -/
  rw [← le_bot_iff]
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    ⊢ LE.le (Submodule.localized₀ p f Bot.bot) Bot.bot
  -/
  rintro _ ⟨_, rfl, s, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    s : Subtype fun x => Membership.mem p x
    ⊢ Membership.mem Bot.bot (IsLocalizedModule.mk' f 0 s)
  -/
  simp only [IsLocalizedModule.mk'_zero, mem_bot]
  /-
    🎉 no goals
  -/


@[simp]
lemma localized'_bot : (⊥ : Submodule R M).localized' S p f = ⊥ :=
                   /-
                     R : Type u_1
                     S : Type u_2
                     M : Type u_3
                     N : Type u_4
                     inst✝¹⁰ : CommSemiring R
                     inst✝⁹ : CommSemiring S
                     inst✝⁸ : AddCommMonoid M
                     inst✝⁷ : AddCommMonoid N
                     inst✝⁶ : Module R M
                     inst✝⁵ : Module R N
                     inst✝⁴ : Algebra R S
                     inst✝³ : Module S N
                     inst✝² : IsScalarTower R S N
                     p : Submonoid R
                     inst✝¹ : IsLocalization p S
                     f : LinearMap (RingHom.id R) M N
                     inst✝ : IsLocalizedModule p f
                     ⊢ Eq ↑(Submodule.localized' S p f Bot.bot) ↑Bot.bot
                   -/
  SetLike.ext' (by apply SetLike.ext'_iff.mp <| Submodule.localized₀_bot p f)
                   /-
                     🎉 no goals
                   -/


@[simp]
lemma localized₀_top : (⊤ : Submodule R M).localized₀ p f = ⊤ := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    ⊢ Eq (Submodule.localized₀ p f Top.top) Top.top
  -/
  rw [← top_le_iff]
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    ⊢ LE.le Top.top (Submodule.localized₀ p f Top.top)
  -/
  rintro x _
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    x : N
    a✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Submodule.localized₀ p f Top.top) x
  -/
  obtain ⟨⟨x, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective p f x
  /-
    case intro.mk
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    x : M
    s : Subtype fun x => Membership.mem p x
    a✝ : Membership.mem Top.top (Function.uncurry (IsLocalizedModule.mk' f) { fst  …
    ⊢ Membership.mem (Submodule.localized₀ p f Top.top) (Function.uncurry (IsLocal …
  -/
  exact ⟨x, trivial, s, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma localized'_top : (⊤ : Submodule R M).localized' S p f = ⊤ :=
                   /-
                     R : Type u_1
                     S : Type u_2
                     M : Type u_3
                     N : Type u_4
                     inst✝¹⁰ : CommSemiring R
                     inst✝⁹ : CommSemiring S
                     inst✝⁸ : AddCommMonoid M
                     inst✝⁷ : AddCommMonoid N
                     inst✝⁶ : Module R M
                     inst✝⁵ : Module R N
                     inst✝⁴ : Algebra R S
                     inst✝³ : Module S N
                     inst✝² : IsScalarTower R S N
                     p : Submonoid R
                     inst✝¹ : IsLocalization p S
                     f : LinearMap (RingHom.id R) M N
                     inst✝ : IsLocalizedModule p f
                     ⊢ Eq ↑(Submodule.localized' S p f Top.top) ↑Top.top
                   -/
  SetLike.ext' (by apply SetLike.ext'_iff.mp <| Submodule.localized₀_top p f)
                   /-
                     🎉 no goals
                   -/


@[simp]
lemma localized'_span (s : Set M) : (span R s).localized' S p f = span S (f '' s) := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    s : Set M
    ⊢ Eq (Submodule.localized' S p f (Submodule.span R s)) (Submodule.span S (Set. …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      ⊢ LE.le (Submodule.localized' S p f (Submodule.span R s)) (Submodule.span S (S …
    -/
  · rintro _ ⟨x, hx, t, rfl⟩
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      ⊢ Membership.mem (Submodule.span S (Set.image (⇑f) s)) (IsLocalizedModule.mk'  …
    -/
    have := IsLocalizedModule.mk'_smul_mk' S f 1 x t 1
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (IsLocalizedModule.mk' f x 1 …
      ⊢ Membership.mem (Submodule.span S (Set.image (⇑f) s)) (IsLocalizedModule.mk'  …
    -/
    simp only [IsLocalizedModule.mk'_one, one_smul, mul_one] at this
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (f x)) (IsLocalizedModule.mk …
      ⊢ Membership.mem (Submodule.span S (Set.image (⇑f) s)) (IsLocalizedModule.mk'  …
    -/
    rw [← this]
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (f x)) (IsLocalizedModule.mk …
      ⊢ Membership.mem (Submodule.span S (Set.image (⇑f) s)) (HSMul.hSMul (IsLocaliz …
    -/
    apply Submodule.smul_mem
    /-
      case a.intro.intro.intro.h
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (f x)) (IsLocalizedModule.mk …
      ⊢ Membership.mem (Submodule.span S (Set.image (⇑f) s)) (f x)
    -/
    rw [← Submodule.restrictScalars_mem R, ← Submodule.mem_comap]
    /-
      case a.intro.intro.intro.h
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (f x)) (IsLocalizedModule.mk …
      ⊢ Membership.mem (Submodule.comap f (Submodule.restrictScalars R (Submodule.sp …
    -/
    refine (show span R s ≤ _ from ?_) hx
    /-
      case a.intro.intro.intro.h
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (f x)) (IsLocalizedModule.mk …
      ⊢ LE.le (Submodule.span R s) (Submodule.comap f (Submodule.restrictScalars R ( …
    -/
    rw [← Submodule.map_le_iff_le_comap, Submodule.map_span]
    /-
      case a.intro.intro.intro.h
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem (Submodule.span R s) x
      t : Subtype fun x => Membership.mem p x
      this : Eq (HSMul.hSMul (IsLocalization.mk' S 1 t) (f x)) (IsLocalizedModule.mk …
      ⊢ LE.le (Submodule.span R (Set.image (⇑f) s)) (Submodule.restrictScalars R (Su …
    -/
    exact span_le_restrictScalars _ _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      ⊢ LE.le (Submodule.span S (Set.image (⇑f) s)) (Submodule.localized' S p f (Sub …
    -/
  · rw [Submodule.span_le, Set.image_subset_iff]
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      ⊢ HasSubset.Subset s (Set.preimage ⇑f ↑(Submodule.localized' S p f (Submodule. …
    -/
    intro x hx
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      s : Set M
      x : M
      hx : Membership.mem s x
      ⊢ Membership.mem (Set.preimage ⇑f ↑(Submodule.localized' S p f (Submodule.span …
    -/
    exact ⟨x, subset_span hx, 1, IsLocalizedModule.mk'_one _ _ _⟩
    /-
      🎉 no goals
    -/


/-- The localization map of a submodule. -/
@[simps!]
                                                                                     /-
                                                                                       R : Type u_1
                                                                                       S : Type u_2
                                                                                       M : Type u_3
                                                                                       N : Type u_4
                                                                                       inst✝¹⁰ : CommSemiring R
                                                                                       inst✝⁹ : CommSemiring S
                                                                                       inst✝⁸ : AddCommMonoid M
                                                                                       inst✝⁷ : AddCommMonoid N
                                                                                       inst✝⁶ : Module R M
                                                                                       inst✝⁵ : Module R N
                                                                                       inst✝⁴ : Algebra R S
                                                                                       inst✝³ : Module S N
                                                                                       inst✝² : IsScalarTower R S N
                                                                                       p : Submonoid R
                                                                                       inst✝¹ : IsLocalization p S
                                                                                       f : LinearMap (RingHom.id R) M N
                                                                                       inst✝ : IsLocalizedModule p f
                                                                                       M' : Submodule R M
                                                                                       x : M
                                                                                       hx : Membership.mem M' x
                                                                                       ⊢ Eq (IsLocalizedModule.mk' f x 1) (f x)
                                                                                     -/
def toLocalized₀ : M' →ₗ[R] M'.localized₀ p f := f.restrict fun x hx ↦ ⟨x, hx, 1, by simp⟩
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- The localization map of a submodule. -/
@[simps!]
def toLocalized' : M' →ₗ[R] M'.localized' S p f := toLocalized₀ p f M'


/-- The localization map of a submodule. -/
abbrev toLocalized : M' →ₗ[R] M'.localized p :=
  M'.toLocalized' (Localization p) p (LocalizedModule.mkLinearMap p M)


instance : IsLocalizedModule p (M'.toLocalized₀ p f) where
  map_units x := by
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      x : Subtype fun x => Membership.mem p x
      ⊢ IsUnit ((algebraMap R (Module.End R (Subtype fun x => Membership.mem (Submod …
    -/
    simp_rw [Module.End_isUnit_iff]
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      x : Subtype fun x => Membership.mem p x
      ⊢ Function.Bijective ⇑((algebraMap R (Module.End R (Subtype fun x => Membershi …
    -/
    constructor
    · exact fun _ _ e ↦ Subtype.ext
        (IsLocalizedModule.smul_injective f x (congr_arg Subtype.val e))
      /-
        case right
        R : Type u_1
        S : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : AddCommMonoid M
        inst✝⁷ : AddCommMonoid N
        inst✝⁶ : Module R M
        inst✝⁵ : Module R N
        inst✝⁴ : Algebra R S
        inst✝³ : Module S N
        inst✝² : IsScalarTower R S N
        p : Submonoid R
        inst✝¹ : IsLocalization p S
        f : LinearMap (RingHom.id R) M N
        inst✝ : IsLocalizedModule p f
        M' : Submodule R M
        x : Subtype fun x => Membership.mem p x
        ⊢ Function.Surjective ⇑((algebraMap R (Module.End R (Subtype fun x => Membersh …
      -/
    · rintro ⟨_, m, hm, s, rfl⟩
      /-
        case right.mk.intro.intro.intro
        R : Type u_1
        S : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : AddCommMonoid M
        inst✝⁷ : AddCommMonoid N
        inst✝⁶ : Module R M
        inst✝⁵ : Module R N
        inst✝⁴ : Algebra R S
        inst✝³ : Module S N
        inst✝² : IsScalarTower R S N
        p : Submonoid R
        inst✝¹ : IsLocalization p S
        f : LinearMap (RingHom.id R) M N
        inst✝ : IsLocalizedModule p f
        M' : Submodule R M
        x : Subtype fun x => Membership.mem p x
        m : M
        hm : Membership.mem M' m
        s : Subtype fun x => Membership.mem p x
        ⊢ Exists fun a => Eq (((algebraMap R (Module.End R (Subtype fun x => Membershi …
      -/
      refine ⟨⟨IsLocalizedModule.mk' f m (s * x), ⟨_, hm, _, rfl⟩⟩, Subtype.ext ?_⟩
      rw [Module.algebraMap_end_apply, SetLike.val_smul_of_tower,
        ← IsLocalizedModule.mk'_smul, ← Submonoid.smul_def, IsLocalizedModule.mk'_cancel_right]
  surj' := by
    /-
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Submodule.localized₀ p f M') x), Exi …
    -/
    rintro ⟨y, x, hx, s, rfl⟩
    /-
      case mk.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M' : Submodule R M
      x : M
      hx : Membership.mem M' x
      s : Subtype fun x => Membership.mem p x
      ⊢ Exists fun x_1 => Eq (HSMul.hSMul x_1.2 ⟨IsLocalizedModule.mk' f x s, ⋯⟩) (( …
    -/
    exact ⟨⟨⟨x, hx⟩, s⟩, by ext; simp⟩
    /-
      🎉 no goals
    -/
  exists_of_eq e := by simpa [Subtype.ext_iff] using
      IsLocalizedModule.exists_of_eq (S := p) (f := f) (congr_arg Subtype.val e)


instance isLocalizedModule : IsLocalizedModule p (M'.toLocalized' S p f) :=
  inferInstanceAs (IsLocalizedModule p (M'.toLocalized₀ p f))


lemma localized₀_le_localized₀_of_smul_le {P Q : Submodule R M} (x : p) (h : x • P ≤ Q) :
    P.localized₀ p f ≤ Q.localized₀ p f := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    P Q : Submodule R M
    x : Subtype fun x => Membership.mem p x
    h : LE.le (HSMul.hSMul x P) Q
    ⊢ LE.le (Submodule.localized₀ p f P) (Submodule.localized₀ p f Q)
  -/
  rintro - ⟨a, ha, r, rfl⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    P Q : Submodule R M
    x : Subtype fun x => Membership.mem p x
    h : LE.le (HSMul.hSMul x P) Q
    a : M
    ha : Membership.mem P a
    r : Subtype fun x => Membership.mem p x
    ⊢ Membership.mem (Submodule.localized₀ p f Q) (IsLocalizedModule.mk' f a r)
  -/
  refine ⟨x • a, h ⟨a, ha, rfl⟩, x * r, ?_⟩
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R M
    inst✝¹ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    P Q : Submodule R M
    x : Subtype fun x => Membership.mem p x
    h : LE.le (HSMul.hSMul x P) Q
    a : M
    ha : Membership.mem P a
    r : Subtype fun x => Membership.mem p x
    ⊢ Eq (IsLocalizedModule.mk' f (HSMul.hSMul x a) (HMul.hMul x r)) (IsLocalizedM …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma localized'_le_localized'_of_smul_le {P Q : Submodule R M} (x : p) (h : x • P ≤ Q) :
    P.localized' S p f ≤ Q.localized' S p f :=
  localized₀_le_localized₀_of_smul_le p f x h


/-- The localization map of a quotient module. -/
def Submodule.toLocalizedQuotient' : M ⧸ M' →ₗ[R] N ⧸ M'.localized' S p f :=
                                                                                          /-
                                                                                            R✝ : Type u_1
                                                                                            S✝ : Type u_2
                                                                                            M✝ : Type u_3
                                                                                            N✝ : Type u_4
                                                                                            inst✝²¹ : CommSemiring R✝
                                                                                            inst✝²⁰ : CommSemiring S✝
                                                                                            inst✝¹⁹ : AddCommMonoid M✝
                                                                                            inst✝¹⁸ : AddCommMonoid N✝
                                                                                            inst✝¹⁷ : Module R✝ M✝
                                                                                            inst✝¹⁶ : Module R✝ N✝
                                                                                            inst✝¹⁵ : Algebra R✝ S✝
                                                                                            inst✝¹⁴ : Module S✝ N✝
                                                                                            inst✝¹³ : IsScalarTower R✝ S✝ N✝
                                                                                            p✝ : Submonoid R✝
                                                                                            inst✝¹² : IsLocalization p✝ S✝
                                                                                            f✝ : LinearMap (RingHom.id R✝) M✝ N✝
                                                                                            inst✝¹¹ : IsLocalizedModule p✝ f✝
                                                                                            M'✝ : Submodule R✝ M✝
                                                                                            R : Type u_5
                                                                                            S : Type u_6
                                                                                            M : Type u_7
                                                                                            N : Type u_8
                                                                                            inst✝¹⁰ : CommRing R
                                                                                            inst✝⁹ : CommRing S
                                                                                            inst✝⁸ : AddCommGroup M
                                                                                            inst✝⁷ : AddCommGroup N
                                                                                            inst✝⁶ : Module R M
                                                                                            inst✝⁵ : Module R N
                                                                                            inst✝⁴ : Algebra R S
                                                                                            inst✝³ : Module S N
                                                                                            inst✝² : IsScalarTower R S N
                                                                                            p : Submonoid R
                                                                                            inst✝¹ : IsLocalization p S
                                                                                            f : LinearMap (RingHom.id R) M N
                                                                                            inst✝ : IsLocalizedModule p f
                                                                                            M' : Submodule R M
                                                                                            x : M
                                                                                            hx : Membership.mem M' x
                                                                                            ⊢ Eq (IsLocalizedModule.mk' f x 1) (f x)
                                                                                          -/
  Submodule.mapQ M' ((M'.localized' S p f).restrictScalars R) f (fun x hx ↦ ⟨x, hx, 1, by simp⟩)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- The localization map of a quotient module. -/
abbrev Submodule.toLocalizedQuotient : M ⧸ M' →ₗ[R] LocalizedModule p M ⧸ M'.localized p :=
  M'.toLocalizedQuotient' (Localization p) p (LocalizedModule.mkLinearMap p M)


@[simp]
lemma Submodule.toLocalizedQuotient'_mk (x : M) :
    M'.toLocalizedQuotient' S p f (Submodule.Quotient.mk x) = Submodule.Quotient.mk (f x) := rfl


open Submodule Submodule.Quotient IsLocalization in
instance IsLocalizedModule.toLocalizedQuotient' (M' : Submodule R M) :
    IsLocalizedModule p (M'.toLocalizedQuotient' S p f) where
  map_units x := by
    refine (Module.End_isUnit_iff _).mpr ⟨fun m n e ↦ ?_, fun m ↦ ⟨(IsLocalization.mk' S 1 x) • m,
        by rw [Module.algebraMap_end_apply, ← smul_assoc, smul_mk'_one, mk'_self', one_smul]⟩⟩
    /-
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      x : Subtype fun x => Membership.mem p x
      m n : HasQuotient.Quotient N (Submodule.localized' S p f M')
      e : Eq (((algebraMap R (Module.End R (HasQuotient.Quotient N (Submodule.locali …
      ⊢ Eq m n
    -/
    obtain ⟨⟨m, rfl⟩, n, rfl⟩ := PProd.mk (mk_surjective _ m) (mk_surjective _ n)
    /-
      case mk.intro.intro
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      x : Subtype fun x => Membership.mem p x
      m n : N
      e : Eq (((algebraMap R (Module.End R (HasQuotient.Quotient N (Submodule.locali …
      ⊢ Eq (Submodule.Quotient.mk m) (Submodule.Quotient.mk n)
    -/
    simp only [Module.algebraMap_end_apply, ← mk_smul, Submodule.Quotient.eq, ← smul_sub] at e
    /-
      case mk.intro.intro
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      x : Subtype fun x => Membership.mem p x
      m n : N
      e : Membership.mem (Submodule.localized' S p f M') (HSMul.hSMul (↑x) (HSub.hSu …
      ⊢ Eq (Submodule.Quotient.mk m) (Submodule.Quotient.mk n)
    -/
    replace e := Submodule.smul_mem _ (IsLocalization.mk' S 1 x) e
    /-
      case mk.intro.intro
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      x : Subtype fun x => Membership.mem p x
      m n : N
      e : Membership.mem (Submodule.localized' S p f M') (HSMul.hSMul (IsLocalizatio …
      ⊢ Eq (Submodule.Quotient.mk m) (Submodule.Quotient.mk n)
    -/
    rwa [smul_comm, ← smul_assoc, smul_mk'_one, mk'_self', one_smul, ← Submodule.Quotient.eq] at e
    /-
      🎉 no goals
    -/
  surj' y := by
    /-
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      y : HasQuotient.Quotient N (Submodule.localized' S p f M')
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 y) ((Submodule.toLocalizedQuotient' S p  …
    -/
    obtain ⟨y, rfl⟩ := mk_surjective _ y
    /-
      case intro
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      y : N
      ⊢ Exists fun x => Eq (HSMul.hSMul x.2 (Submodule.Quotient.mk y)) ((Submodule.t …
    -/
    obtain ⟨⟨y, s⟩, rfl⟩ := IsLocalizedModule.mk'_surjective p f y
    exact ⟨⟨Submodule.Quotient.mk y, s⟩,
      by simp only [Function.uncurry_apply_pair, toLocalizedQuotient'_mk, ← mk_smul, mk'_cancel']⟩
  exists_of_eq {m n} e := by
    /-
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      m n : HasQuotient.Quotient M M'
      e : Eq ((Submodule.toLocalizedQuotient' S p f M') m) ((Submodule.toLocalizedQu …
      ⊢ Exists fun c => Eq (HSMul.hSMul c m) (HSMul.hSMul c n)
    -/
    obtain ⟨⟨m, rfl⟩, n, rfl⟩ := PProd.mk (mk_surjective _ m) (mk_surjective _ n)
    /-
      case mk.intro.intro
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      m n : M
      e : Eq ((Submodule.toLocalizedQuotient' S p f M') (Submodule.Quotient.mk m)) ( …
      ⊢ Exists fun c => Eq (HSMul.hSMul c (Submodule.Quotient.mk m)) (HSMul.hSMul c  …
    -/
    obtain ⟨x, hx, s, hs⟩ : f (m - n) ∈ _ := by simpa [Submodule.Quotient.eq] using e
    /-
      case mk.intro.intro.intro.intro.intro
      R✝ : Type u_1
      S✝ : Type u_2
      M✝ : Type u_3
      N✝ : Type u_4
      inst✝²¹ : CommSemiring R✝
      inst✝²⁰ : CommSemiring S✝
      inst✝¹⁹ : AddCommMonoid M✝
      inst✝¹⁸ : AddCommMonoid N✝
      inst✝¹⁷ : Module R✝ M✝
      inst✝¹⁶ : Module R✝ N✝
      inst✝¹⁵ : Algebra R✝ S✝
      inst✝¹⁴ : Module S✝ N✝
      inst✝¹³ : IsScalarTower R✝ S✝ N✝
      p✝ : Submonoid R✝
      inst✝¹² : IsLocalization p✝ S✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      inst✝¹¹ : IsLocalizedModule p✝ f✝
      M'✝¹ : Submodule R✝ M✝
      R : Type u_5
      S : Type u_6
      M : Type u_7
      N : Type u_8
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      M'✝ M' : Submodule R M
      m n : M
      e : Eq ((Submodule.toLocalizedQuotient' S p f M') (Submodule.Quotient.mk m)) ( …
      x : M
      hx : Membership.mem M' x
      s : Subtype fun x => Membership.mem p x
      hs : Eq (IsLocalizedModule.mk' f x s) (f (HSub.hSub m n))
      ⊢ Exists fun c => Eq (HSMul.hSMul c (Submodule.Quotient.mk m)) (HSMul.hSMul c  …
    -/
    obtain ⟨c, hc⟩ := exists_of_eq (S := p) (show f (s • (m - n)) = f x by simp [-map_sub, ← hs])
    exact ⟨c * s, by simpa only [← Quotient.mk_smul, Submodule.Quotient.eq,
      ← smul_sub, mul_smul, hc] using M'.smul_mem c hx⟩


instance (M' : Submodule R M) : IsLocalizedModule p (M'.toLocalizedQuotient p) :=
  IsLocalizedModule.toLocalizedQuotient' _ _ _ _


lemma ker_localizedMap_eq_localized₀_ker (g : M →ₗ[R] P) :
    ker (map p f f' g) = (ker g).localized₀ p f := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝⁵ : IsLocalizedModule p f
    P : Type u_5
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R P
    Q : Type u_6
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    f' : LinearMap (RingHom.id R) P Q
    inst✝ : IsLocalizedModule p f'
    g : LinearMap (RingHom.id R) M P
    ⊢ Eq (LinearMap.ker ((IsLocalizedModule.map p f f') g)) (Submodule.localized₀  …
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝⁵ : IsLocalizedModule p f
    P : Type u_5
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R P
    Q : Type u_6
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    f' : LinearMap (RingHom.id R) P Q
    inst✝ : IsLocalizedModule p f'
    g : LinearMap (RingHom.id R) M P
    x : N
    ⊢ Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.map p f f') g)) x) (M …
  -/
  simp only [Submodule.mem_localized₀, mem_ker]
  /-
    case h
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝⁵ : IsLocalizedModule p f
    P : Type u_5
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R P
    Q : Type u_6
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    f' : LinearMap (RingHom.id R) P Q
    inst✝ : IsLocalizedModule p f'
    g : LinearMap (RingHom.id R) M P
    x : N
    ⊢ Iff (Eq (((IsLocalizedModule.map p f f') g) x) 0) (Exists fun m => And (Eq ( …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case h.refine_1
      R : Type u_1
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      p : Submonoid R
      f : LinearMap (RingHom.id R) M N
      inst✝⁵ : IsLocalizedModule p f
      P : Type u_5
      inst✝⁴ : AddCommMonoid P
      inst✝³ : Module R P
      Q : Type u_6
      inst✝² : AddCommMonoid Q
      inst✝¹ : Module R Q
      f' : LinearMap (RingHom.id R) P Q
      inst✝ : IsLocalizedModule p f'
      g : LinearMap (RingHom.id R) M P
      x : N
      h : Eq (((IsLocalizedModule.map p f f') g) x) 0
      ⊢ Exists fun m => And (Eq (g m) 0) (Exists fun s => Eq (IsLocalizedModule.mk'  …
    -/
  · obtain ⟨⟨a, b⟩, rfl⟩ := IsLocalizedModule.mk'_surjective p f x
    /-
      case h.refine_1.intro.mk
      R : Type u_1
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      p : Submonoid R
      f : LinearMap (RingHom.id R) M N
      inst✝⁵ : IsLocalizedModule p f
      P : Type u_5
      inst✝⁴ : AddCommMonoid P
      inst✝³ : Module R P
      Q : Type u_6
      inst✝² : AddCommMonoid Q
      inst✝¹ : Module R Q
      f' : LinearMap (RingHom.id R) P Q
      inst✝ : IsLocalizedModule p f'
      g : LinearMap (RingHom.id R) M P
      a : M
      b : Subtype fun x => Membership.mem p x
      h : Eq (((IsLocalizedModule.map p f f') g) (Function.uncurry (IsLocalizedModul …
      ⊢ Exists fun m => And (Eq (g m) 0) (Exists fun s => Eq (IsLocalizedModule.mk'  …
    -/
    simp only [Function.uncurry_apply_pair, map_mk', mk'_eq_zero, eq_zero_iff p f'] at h
    /-
      case h.refine_1.intro.mk
      R : Type u_1
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      p : Submonoid R
      f : LinearMap (RingHom.id R) M N
      inst✝⁵ : IsLocalizedModule p f
      P : Type u_5
      inst✝⁴ : AddCommMonoid P
      inst✝³ : Module R P
      Q : Type u_6
      inst✝² : AddCommMonoid Q
      inst✝¹ : Module R Q
      f' : LinearMap (RingHom.id R) P Q
      inst✝ : IsLocalizedModule p f'
      g : LinearMap (RingHom.id R) M P
      a : M
      b : Subtype fun x => Membership.mem p x
      h : Exists fun s' => Eq (HSMul.hSMul s' (g a)) 0
      ⊢ Exists fun m => And (Eq (g m) 0) (Exists fun s => Eq (IsLocalizedModule.mk'  …
    -/
    obtain ⟨c, hc⟩ := h
    /-
      case h.refine_1.intro.mk.intro
      R : Type u_1
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      p : Submonoid R
      f : LinearMap (RingHom.id R) M N
      inst✝⁵ : IsLocalizedModule p f
      P : Type u_5
      inst✝⁴ : AddCommMonoid P
      inst✝³ : Module R P
      Q : Type u_6
      inst✝² : AddCommMonoid Q
      inst✝¹ : Module R Q
      f' : LinearMap (RingHom.id R) P Q
      inst✝ : IsLocalizedModule p f'
      g : LinearMap (RingHom.id R) M P
      a : M
      b c : Subtype fun x => Membership.mem p x
      hc : Eq (HSMul.hSMul c (g a)) 0
      ⊢ Exists fun m => And (Eq (g m) 0) (Exists fun s => Eq (IsLocalizedModule.mk'  …
    -/
    refine ⟨c • a, by simpa, c * b, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      p : Submonoid R
      f : LinearMap (RingHom.id R) M N
      inst✝⁵ : IsLocalizedModule p f
      P : Type u_5
      inst✝⁴ : AddCommMonoid P
      inst✝³ : Module R P
      Q : Type u_6
      inst✝² : AddCommMonoid Q
      inst✝¹ : Module R Q
      f' : LinearMap (RingHom.id R) P Q
      inst✝ : IsLocalizedModule p f'
      g : LinearMap (RingHom.id R) M P
      x : N
      ⊢ (Exists fun m => And (Eq (g m) 0) (Exists fun s => Eq (IsLocalizedModule.mk' …
    -/
  · rintro ⟨m, hm, a, ha, rfl⟩
    /-
      case h.refine_2.intro.intro.intro.refl
      R : Type u_1
      M : Type u_3
      N : Type u_4
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : AddCommMonoid M
      inst✝⁸ : AddCommMonoid N
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      p : Submonoid R
      f : LinearMap (RingHom.id R) M N
      inst✝⁵ : IsLocalizedModule p f
      P : Type u_5
      inst✝⁴ : AddCommMonoid P
      inst✝³ : Module R P
      Q : Type u_6
      inst✝² : AddCommMonoid Q
      inst✝¹ : Module R Q
      f' : LinearMap (RingHom.id R) P Q
      inst✝ : IsLocalizedModule p f'
      g : LinearMap (RingHom.id R) M P
      m : M
      hm : Eq (g m) 0
      a : Subtype fun x => Membership.mem p x
      ⊢ Eq (((IsLocalizedModule.map p f f') g) (IsLocalizedModule.mk' f m a)) 0
    -/
    simp [IsLocalizedModule.map_mk', hm]
    /-
      🎉 no goals
    -/


lemma localized'_ker_eq_ker_localizedMap (g : M →ₗ[R] P) :
    (ker g).localized' S p f = ker ((map p f f' g).extendScalarsOfIsLocalization p S) :=
                  /-
                    R : Type u_1
                    S : Type u_2
                    M : Type u_3
                    N : Type u_4
                    inst✝¹⁷ : CommSemiring R
                    inst✝¹⁶ : CommSemiring S
                    inst✝¹⁵ : AddCommMonoid M
                    inst✝¹⁴ : AddCommMonoid N
                    inst✝¹³ : Module R M
                    inst✝¹² : Module R N
                    inst✝¹¹ : Algebra R S
                    inst✝¹⁰ : Module S N
                    inst✝⁹ : IsScalarTower R S N
                    p : Submonoid R
                    inst✝⁸ : IsLocalization p S
                    f : LinearMap (RingHom.id R) M N
                    inst✝⁷ : IsLocalizedModule p f
                    P : Type u_5
                    inst✝⁶ : AddCommMonoid P
                    inst✝⁵ : Module R P
                    Q : Type u_6
                    inst✝⁴ : AddCommMonoid Q
                    inst✝³ : Module R Q
                    inst✝² : Module S Q
                    inst✝¹ : IsScalarTower R S Q
                    f' : LinearMap (RingHom.id R) P Q
                    inst✝ : IsLocalizedModule p f'
                    g : LinearMap (RingHom.id R) M P
                    ⊢ ∀ (x : N), Iff (Membership.mem (Submodule.localized' S p f (LinearMap.ker g) …
                  -/
  SetLike.ext (by apply SetLike.ext_iff.mp (f.ker_localizedMap_eq_localized₀_ker p f' g).symm)
                  /-
                    🎉 no goals
                  -/


lemma ker_localizedMap_eq_localized'_ker (g : M →ₗ[R] P) :
    ker (map p f f' g) = ((ker g).localized' S p f).restrictScalars _ := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : AddCommMonoid M
    inst✝¹⁴ : AddCommMonoid N
    inst✝¹³ : Module R M
    inst✝¹² : Module R N
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Module S N
    inst✝⁹ : IsScalarTower R S N
    p : Submonoid R
    inst✝⁸ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝⁷ : IsLocalizedModule p f
    P : Type u_5
    inst✝⁶ : AddCommMonoid P
    inst✝⁵ : Module R P
    Q : Type u_6
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R Q
    inst✝² : Module S Q
    inst✝¹ : IsScalarTower R S Q
    f' : LinearMap (RingHom.id R) P Q
    inst✝ : IsLocalizedModule p f'
    g : LinearMap (RingHom.id R) M P
    ⊢ Eq (LinearMap.ker ((IsLocalizedModule.map p f f') g)) (Submodule.restrictSca …
  -/
  ext
  /-
    case h
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : AddCommMonoid M
    inst✝¹⁴ : AddCommMonoid N
    inst✝¹³ : Module R M
    inst✝¹² : Module R N
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Module S N
    inst✝⁹ : IsScalarTower R S N
    p : Submonoid R
    inst✝⁸ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝⁷ : IsLocalizedModule p f
    P : Type u_5
    inst✝⁶ : AddCommMonoid P
    inst✝⁵ : Module R P
    Q : Type u_6
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R Q
    inst✝² : Module S Q
    inst✝¹ : IsScalarTower R S Q
    f' : LinearMap (RingHom.id R) P Q
    inst✝ : IsLocalizedModule p f'
    g : LinearMap (RingHom.id R) M P
    x✝ : N
    ⊢ Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.map p f f') g)) x✝) ( …
  -/
  simp [localized'_ker_eq_ker_localizedMap S p f f']
  /-
    🎉 no goals
  -/


/--
The canonical map from the kernel of `g` to the kernel of `g` localized at a submonoid.

This is a localization map by `LinearMap.toKerLocalized_isLocalizedModule`.
-/
@[simps!]
noncomputable def toKerIsLocalized (g : M →ₗ[R] P) :
    ker g →ₗ[R] ker (map p f f' g) :=
                            /-
                              R : Type u_1
                              S : Type u_2
                              M : Type u_3
                              N : Type u_4
                              inst✝¹⁷ : CommSemiring R
                              inst✝¹⁶ : CommSemiring S
                              inst✝¹⁵ : AddCommMonoid M
                              inst✝¹⁴ : AddCommMonoid N
                              inst✝¹³ : Module R M
                              inst✝¹² : Module R N
                              inst✝¹¹ : Algebra R S
                              inst✝¹⁰ : Module S N
                              inst✝⁹ : IsScalarTower R S N
                              p : Submonoid R
                              inst✝⁸ : IsLocalization p S
                              f : LinearMap (RingHom.id R) M N
                              inst✝⁷ : IsLocalizedModule p f
                              M' : Submodule R M
                              P : Type u_5
                              inst✝⁶ : AddCommMonoid P
                              inst✝⁵ : Module R P
                              Q : Type u_6
                              inst✝⁴ : AddCommMonoid Q
                              inst✝³ : Module R Q
                              inst✝² : Module S Q
                              inst✝¹ : IsScalarTower R S Q
                              f' : LinearMap (RingHom.id R) P Q
                              inst✝ : IsLocalizedModule p f'
                              g : LinearMap (RingHom.id R) M P
                              x : M
                              hx : Membership.mem (LinearMap.ker g) x
                              ⊢ Membership.mem (LinearMap.ker ((IsLocalizedModule.map p f f') g)) (f x)
                            -/
  f.restrict (fun x hx ↦ by simp [mem_ker, mem_ker.mp hx])
                            /-
                              🎉 no goals
                            -/


include S in
/-- The canonical map to the kernel of the localization of `g` is localizing.
In other words, localization commutes with kernels. -/
lemma toKerLocalized_isLocalizedModule (g : M →ₗ[R] P) :
    IsLocalizedModule p (toKerIsLocalized p f f' g) :=
  let e : Submodule.localized' S p f (ker g) ≃ₗ[S]
      ker ((map p f f' g).extendScalarsOfIsLocalization p S) :=
    LinearEquiv.ofEq _ _ (localized'_ker_eq_ker_localizedMap S p f f' g)
  IsLocalizedModule.of_linearEquiv p (Submodule.toLocalized' S p f (ker g)) (e.restrictScalars R)


lemma range_localizedMap_eq_localized₀_range (g : M →ₗ[R] P) :
    range (map p f f' g) = (range g).localized₀ p f' := by
  /-
    R : Type u_1
    M : Type u_3
    N : Type u_4
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    p : Submonoid R
    f : LinearMap (RingHom.id R) M N
    inst✝⁵ : IsLocalizedModule p f
    P : Type u_5
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R P
    Q : Type u_6
    inst✝² : AddCommMonoid Q
    inst✝¹ : Module R Q
    f' : LinearMap (RingHom.id R) P Q
    inst✝ : IsLocalizedModule p f'
    g : LinearMap (RingHom.id R) M P
    ⊢ Eq (LinearMap.range ((IsLocalizedModule.map p f f') g)) (Submodule.localized …
  -/
  ext; simp [mem_localized₀, mem_range, (mk'_surjective p f).exists]
       /-
         🎉 no goals
       -/


/-- Localization commutes with ranges. -/
lemma localized'_range_eq_range_localizedMap (g : M →ₗ[R] P) :
    (range g).localized' S p f' = range ((map p f f' g).extendScalarsOfIsLocalization p S) :=
                  /-
                    R : Type u_1
                    S : Type u_2
                    M : Type u_3
                    N : Type u_4
                    inst✝¹⁷ : CommSemiring R
                    inst✝¹⁶ : CommSemiring S
                    inst✝¹⁵ : AddCommMonoid M
                    inst✝¹⁴ : AddCommMonoid N
                    inst✝¹³ : Module R M
                    inst✝¹² : Module R N
                    inst✝¹¹ : Algebra R S
                    inst✝¹⁰ : Module S N
                    inst✝⁹ : IsScalarTower R S N
                    p : Submonoid R
                    inst✝⁸ : IsLocalization p S
                    f : LinearMap (RingHom.id R) M N
                    inst✝⁷ : IsLocalizedModule p f
                    P : Type u_5
                    inst✝⁶ : AddCommMonoid P
                    inst✝⁵ : Module R P
                    Q : Type u_6
                    inst✝⁴ : AddCommMonoid Q
                    inst✝³ : Module R Q
                    inst✝² : Module S Q
                    inst✝¹ : IsScalarTower R S Q
                    f' : LinearMap (RingHom.id R) P Q
                    inst✝ : IsLocalizedModule p f'
                    g : LinearMap (RingHom.id R) M P
                    ⊢ ∀ (x : Q), Iff (Membership.mem (Submodule.localized' S p f' (LinearMap.range …
                  -/
  SetLike.ext (by apply SetLike.ext_iff.mp (f.range_localizedMap_eq_localized₀_range p f' g).symm)
                  /-
                    🎉 no goals
                  -/


