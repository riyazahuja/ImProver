/-- Localizing wrt `M ⊆ R` and then wrt `N ⊆ S = M⁻¹R` is equal to the localization of `R` wrt this
module. See `localization_localization_isLocalization`.
-/
@[nolint unusedArguments]
def localizationLocalizationSubmodule : Submonoid R :=
  (N ⊔ M.map (algebraMap R S)).comap (algebraMap R S)


@[simp]
theorem mem_localizationLocalizationSubmodule {x : R} :
    x ∈ localizationLocalizationSubmodule M N ↔
      ∃ (y : N) (z : M), algebraMap R S x = y * algebraMap R S z := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    N : Submonoid S
    x : R
    ⊢ Iff (Membership.mem (IsLocalization.localizationLocalizationSubmodule M N) x …
  -/
  rw [localizationLocalizationSubmodule, Submonoid.mem_comap, Submonoid.mem_sup]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    N : Submonoid S
    x : R
    ⊢ Iff (Exists fun y => And (Membership.mem N y) (Exists fun z => And (Membersh …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      N : Submonoid S
      x : R
      ⊢ (Exists fun y => And (Membership.mem N y) (Exists fun z => And (Membership.m …
    -/
  · rintro ⟨y, hy, _, ⟨z, hz, rfl⟩, e⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝² : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      N : Submonoid S
      x : R
      y : S
      hy : Membership.mem N y
      z : R
      hz : Membership.mem (↑M) z
      e : Eq (HMul.hMul y ((algebraMap R S) z)) ((algebraMap R S) x)
      ⊢ Exists fun y => Exists fun z => Eq ((algebraMap R S) x) (HMul.hMul (↑y) ((al …
    -/
    exact ⟨⟨y, hy⟩, ⟨z, hz⟩, e.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      N : Submonoid S
      x : R
      ⊢ (Exists fun y => Exists fun z => Eq ((algebraMap R S) x) (HMul.hMul (↑y) ((a …
    -/
  · rintro ⟨y, z, e⟩
    /-
      case mpr.intro.intro
      R : Type u_1
      inst✝² : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝¹ : CommSemiring S
      inst✝ : Algebra R S
      N : Submonoid S
      x : R
      y : Subtype fun x => Membership.mem N x
      z : Subtype fun x => Membership.mem M x
      e : Eq ((algebraMap R S) x) (HMul.hMul (↑y) ((algebraMap R S) ↑z))
      ⊢ Exists fun y => And (Membership.mem N y) (Exists fun z => And (Membership.me …
    -/
    exact ⟨y, y.prop, _, ⟨z, z.prop, rfl⟩, e.symm⟩
    /-
      🎉 no goals
    -/


theorem localization_localization_map_units [IsLocalization N T]
    (y : localizationLocalizationSubmodule M N) : IsUnit (algebraMap R T y) := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    y : Subtype fun x => Membership.mem (IsLocalization.localizationLocalizationSu …
    ⊢ IsUnit ((algebraMap R T) ↑y)
  -/
  obtain ⟨y', z, eq⟩ := mem_localizationLocalizationSubmodule.mp y.prop
  /-
    case intro.intro
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    y : Subtype fun x => Membership.mem (IsLocalization.localizationLocalizationSu …
    y' : Subtype fun x => Membership.mem N x
    z : Subtype fun x => Membership.mem M x
    eq : Eq ((algebraMap R S) ↑y) (HMul.hMul (↑y') ((algebraMap R S) ↑z))
    ⊢ IsUnit ((algebraMap R T) ↑y)
  -/
  rw [IsScalarTower.algebraMap_apply R S T, eq, RingHom.map_mul, IsUnit.mul_iff]
  /-
    case intro.intro
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    y : Subtype fun x => Membership.mem (IsLocalization.localizationLocalizationSu …
    y' : Subtype fun x => Membership.mem N x
    z : Subtype fun x => Membership.mem M x
    eq : Eq ((algebraMap R S) ↑y) (HMul.hMul (↑y') ((algebraMap R S) ↑z))
    ⊢ And (IsUnit ((algebraMap S T) ↑y')) (IsUnit ((algebraMap S T) ((algebraMap R …
  -/
  exact ⟨IsLocalization.map_units T y', (IsLocalization.map_units _ z).map (algebraMap S T)⟩
  /-
    🎉 no goals
  -/


theorem localization_localization_surj [IsLocalization N T] (x : T) :
    ∃ y : R × localizationLocalizationSubmodule M N,
        x * algebraMap R T y.2 = algebraMap R T y.1 := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x : T
    ⊢ Exists fun y => Eq (HMul.hMul x ((algebraMap R T) ↑y.2)) ((algebraMap R T) y …
  -/
  rcases IsLocalization.surj N x with ⟨⟨y, s⟩, eq₁⟩
  -- x = y / s
  /-
    case intro.mk
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x : T
    y : S
    s : Subtype fun x => Membership.mem N x
    eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑{ fst := y, snd := s }.2)) ((algebraM …
    ⊢ Exists fun y => Eq (HMul.hMul x ((algebraMap R T) ↑y.2)) ((algebraMap R T) y …
  -/
  rcases IsLocalization.surj M y with ⟨⟨z, t⟩, eq₂⟩
  -- y = z / t
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x : T
    y : S
    s : Subtype fun x => Membership.mem N x
    eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑{ fst := y, snd := s }.2)) ((algebraM …
    z : R
    t : Subtype fun x => Membership.mem M x
    eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := z, snd := t }.2)) ((algebraM …
    ⊢ Exists fun y => Eq (HMul.hMul x ((algebraMap R T) ↑y.2)) ((algebraMap R T) y …
  -/
  rcases IsLocalization.surj M (s : S) with ⟨⟨z', t'⟩, eq₃⟩
  -- s = z' / t'
  /-
    case intro.mk.intro.mk.intro.mk
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x : T
    y : S
    s : Subtype fun x => Membership.mem N x
    eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑{ fst := y, snd := s }.2)) ((algebraM …
    z : R
    t : Subtype fun x => Membership.mem M x
    eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := z, snd := t }.2)) ((algebraM …
    z' : R
    t' : Subtype fun x => Membership.mem M x
    eq₃ : Eq (HMul.hMul (↑s) ((algebraMap R S) ↑{ fst := z', snd := t' }.2)) ((alg …
    ⊢ Exists fun y => Eq (HMul.hMul x ((algebraMap R T) ↑y.2)) ((algebraMap R T) y …
  -/
  dsimp only at eq₁ eq₂ eq₃
  /-
    case intro.mk.intro.mk.intro.mk
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x : T
    y : S
    s : Subtype fun x => Membership.mem N x
    eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑s)) ((algebraMap S T) y)
    z : R
    t : Subtype fun x => Membership.mem M x
    eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑t)) ((algebraMap R S) z)
    z' : R
    t' : Subtype fun x => Membership.mem M x
    eq₃ : Eq (HMul.hMul (↑s) ((algebraMap R S) ↑t')) ((algebraMap R S) z')
    ⊢ Exists fun y => Eq (HMul.hMul x ((algebraMap R T) ↑y.2)) ((algebraMap R T) y …
  -/
  refine ⟨⟨z * t', z' * t, ?_⟩, ?_⟩ -- x = y / s = (z * t') / (z' * t)
    /-
      case intro.mk.intro.mk.intro.mk.refine_1
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      N : Submonoid S
      T : Type u_3
      inst✝⁵ : CommSemiring T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsLocalization N T
      x : T
      y : S
      s : Subtype fun x => Membership.mem N x
      eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑s)) ((algebraMap S T) y)
      z : R
      t : Subtype fun x => Membership.mem M x
      eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑t)) ((algebraMap R S) z)
      z' : R
      t' : Subtype fun x => Membership.mem M x
      eq₃ : Eq (HMul.hMul (↑s) ((algebraMap R S) ↑t')) ((algebraMap R S) z')
      ⊢ Membership.mem (IsLocalization.localizationLocalizationSubmodule M N) (HMul. …
    -/
  · rw [mem_localizationLocalizationSubmodule]
    /-
      case intro.mk.intro.mk.intro.mk.refine_1
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      N : Submonoid S
      T : Type u_3
      inst✝⁵ : CommSemiring T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsLocalization N T
      x : T
      y : S
      s : Subtype fun x => Membership.mem N x
      eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑s)) ((algebraMap S T) y)
      z : R
      t : Subtype fun x => Membership.mem M x
      eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑t)) ((algebraMap R S) z)
      z' : R
      t' : Subtype fun x => Membership.mem M x
      eq₃ : Eq (HMul.hMul (↑s) ((algebraMap R S) ↑t')) ((algebraMap R S) z')
      ⊢ Exists fun y => Exists fun z => Eq ((algebraMap R S) (HMul.hMul z' ↑t)) (HMu …
    -/
    refine ⟨s, t * t', ?_⟩
    /-
      case intro.mk.intro.mk.intro.mk.refine_1
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      N : Submonoid S
      T : Type u_3
      inst✝⁵ : CommSemiring T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsLocalization N T
      x : T
      y : S
      s : Subtype fun x => Membership.mem N x
      eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑s)) ((algebraMap S T) y)
      z : R
      t : Subtype fun x => Membership.mem M x
      eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑t)) ((algebraMap R S) z)
      z' : R
      t' : Subtype fun x => Membership.mem M x
      eq₃ : Eq (HMul.hMul (↑s) ((algebraMap R S) ↑t')) ((algebraMap R S) z')
      ⊢ Eq ((algebraMap R S) (HMul.hMul z' ↑t)) (HMul.hMul (↑s) ((algebraMap R S) ↑( …
    -/
    rw [RingHom.map_mul, ← eq₃, mul_assoc, ← RingHom.map_mul, mul_comm t, Submonoid.coe_mul]
    /-
      🎉 no goals
    -/
  · simp only [Subtype.coe_mk, RingHom.map_mul, IsScalarTower.algebraMap_apply R S T, ← eq₃, ← eq₂,
      ← eq₁]
    /-
      case intro.mk.intro.mk.intro.mk.refine_2
      R : Type u_1
      inst✝⁸ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      N : Submonoid S
      T : Type u_3
      inst✝⁵ : CommSemiring T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsLocalization N T
      x : T
      y : S
      s : Subtype fun x => Membership.mem N x
      eq₁ : Eq (HMul.hMul x ((algebraMap S T) ↑s)) ((algebraMap S T) y)
      z : R
      t : Subtype fun x => Membership.mem M x
      eq₂ : Eq (HMul.hMul y ((algebraMap R S) ↑t)) ((algebraMap R S) z)
      z' : R
      t' : Subtype fun x => Membership.mem M x
      eq₃ : Eq (HMul.hMul (↑s) ((algebraMap R S) ↑t')) ((algebraMap R S) z')
      ⊢ Eq (HMul.hMul x (HMul.hMul (HMul.hMul ((algebraMap S T) ↑s) ((algebraMap S T …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem localization_localization_exists_of_eq [IsLocalization N T] (x y : R) :
    algebraMap R T x = algebraMap R T y →
      ∃ c : localizationLocalizationSubmodule M N, ↑c * x = ↑c * y := by
  rw [IsScalarTower.algebraMap_apply R S T, IsScalarTower.algebraMap_apply R S T,
    IsLocalization.eq_iff_exists N T]
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x y : R
    ⊢ (Exists fun c => Eq (HMul.hMul (↑c) ((algebraMap R S) x)) (HMul.hMul (↑c) (( …
  -/
  rintro ⟨z, eq₁⟩
  /-
    case intro
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x y : R
    z : Subtype fun x => Membership.mem N x
    eq₁ : Eq (HMul.hMul (↑z) ((algebraMap R S) x)) (HMul.hMul (↑z) ((algebraMap R  …
    ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
  -/
  rcases IsLocalization.surj M (z : S) with ⟨⟨z', s⟩, eq₂⟩
  /-
    case intro.intro.mk
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    x y : R
    z : Subtype fun x => Membership.mem N x
    eq₁ : Eq (HMul.hMul (↑z) ((algebraMap R S) x)) (HMul.hMul (↑z) ((algebraMap R  …
    z' : R
    s : Subtype fun x => Membership.mem M x
    eq₂ : Eq (HMul.hMul (↑z) ((algebraMap R S) ↑{ fst := z', snd := s }.2)) ((alge …
    ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
  -/
  dsimp only at eq₂
  suffices (algebraMap R S) (x * z' : R) = (algebraMap R S) (y * z') by
    obtain ⟨c, eq₃ : ↑c * (x * z') = ↑c * (y * z')⟩ := (IsLocalization.eq_iff_exists M S).mp this
    refine ⟨⟨c * z', ?_⟩, ?_⟩
    · rw [mem_localizationLocalizationSubmodule]
      refine ⟨z, c * s, ?_⟩
      rw [map_mul, ← eq₂, Submonoid.coe_mul, map_mul, mul_left_comm]
    · rwa [mul_comm _ z', mul_comm _ z', ← mul_assoc, ← mul_assoc] at eq₃
  rw [map_mul, map_mul, ← eq₂, ← mul_assoc, ← mul_assoc, mul_comm _ (z : S), eq₁,
    mul_comm _ (z : S)]


/-- Given submodules `M ⊆ R` and `N ⊆ S = M⁻¹R`, with `f : R →+* S` the localization map, we have
`N ⁻¹ S = T = (f⁻¹ (N • f(M))) ⁻¹ R`. I.e., the localization of a localization is a localization.
-/
theorem localization_localization_isLocalization [IsLocalization N T] :
    IsLocalization (localizationLocalizationSubmodule M N) T :=
  { map_units' := localization_localization_map_units M N T
    surj' := localization_localization_surj M N T
    exists_of_eq := localization_localization_exists_of_eq M N T _ _ }


include M in
/-- Given submodules `M ⊆ R` and `N ⊆ S = M⁻¹R`, with `f : R →+* S` the localization map, if
`N` contains all the units of `S`, then `N ⁻¹ S = T = (f⁻¹ N) ⁻¹ R`. I.e., the localization of a
localization is a localization.
-/
theorem localization_localization_isLocalization_of_has_all_units [IsLocalization N T]
    (H : ∀ x : S, IsUnit x → x ∈ N) : IsLocalization (N.comap (algebraMap R S)) T := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    ⊢ IsLocalization (Submonoid.comap (algebraMap R S) N) T
  -/
  convert localization_localization_isLocalization M N T using 1
  /-
    case h.e'_3
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    ⊢ Eq (Submonoid.comap (algebraMap R S) N) (IsLocalization.localizationLocaliza …
  -/
  dsimp [localizationLocalizationSubmodule]
  /-
    case h.e'_3
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    ⊢ Eq (Submonoid.comap (algebraMap R S) N) (Submonoid.comap (algebraMap R S) (M …
  -/
  congr
  /-
    case h.e'_3.e_S
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    ⊢ Eq N (Max.max N (Submonoid.map (algebraMap R S) M))
  -/
  symm
  /-
    case h.e'_3.e_S
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    ⊢ Eq (Max.max N (Submonoid.map (algebraMap R S) M)) N
  -/
  rw [sup_eq_left]
  /-
    case h.e'_3.e_S
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    ⊢ LE.le (Submonoid.map (algebraMap R S) M) N
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case h.e'_3.e_S.intro.intro
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Submonoid S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsLocalization N T
    H : ∀ (x : S), IsUnit x → Membership.mem N x
    x : R
    hx : Membership.mem (↑M) x
    ⊢ Membership.mem N ((algebraMap R S) x)
  -/
  exact H _ (IsLocalization.map_units _ ⟨x, hx⟩)
  /-
    🎉 no goals
  -/


include M in
/--
Given a submodule `M ⊆ R` and a prime ideal `p` of `S = M⁻¹R`, with `f : R →+* S` the localization
map, then `T = Sₚ` is the localization of `R` at `f⁻¹(p)`.
-/
theorem isLocalization_isLocalization_atPrime_isLocalization (p : Ideal S) [Hp : p.IsPrime]
    [IsLocalization.AtPrime T p] : IsLocalization.AtPrime T (p.comap (algebraMap R S)) := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    p : Ideal S
    Hp : p.IsPrime
    inst✝ : IsLocalization.AtPrime T p
    ⊢ IsLocalization.AtPrime T (Ideal.comap (algebraMap R S) p)
  -/
  apply localization_localization_isLocalization_of_has_all_units M p.primeCompl T
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    p : Ideal S
    Hp : p.IsPrime
    inst✝ : IsLocalization.AtPrime T p
    ⊢ ∀ (x : S), IsUnit x → Membership.mem p.primeCompl x
  -/
  intro x hx hx'
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommSemiring T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    p : Ideal S
    Hp : p.IsPrime
    inst✝ : IsLocalization.AtPrime T p
    x : S
    hx : IsUnit x
    hx' : Membership.mem (↑p) x
    ⊢ False
  -/
  exact (Hp.1 : ¬_) (p.eq_top_of_isUnit_mem hx' hx)
  /-
    🎉 no goals
  -/


instance (p : Ideal (Localization M)) [p.IsPrime] : Algebra R (Localization.AtPrime p) :=
  inferInstance


instance (p : Ideal (Localization M)) [p.IsPrime] :
    IsScalarTower R (Localization M) (Localization.AtPrime p) :=
  IsScalarTower.of_algebraMap_eq' rfl


instance localization_localization_atPrime_is_localization (p : Ideal (Localization M))
    [p.IsPrime] : IsLocalization.AtPrime (Localization.AtPrime p) (p.comap (algebraMap R _)) :=
  isLocalization_isLocalization_atPrime_isLocalization M _ _


/-- Given a submodule `M ⊆ R` and a prime ideal `p` of `M⁻¹R`, with `f : R →+* S` the localization
map, then `(M⁻¹R)ₚ` is isomorphic (as an `R`-algebra) to the localization of `R` at `f⁻¹(p)`.
-/
noncomputable def localizationLocalizationAtPrimeIsoLocalization (p : Ideal (Localization M))
    [p.IsPrime] :
    Localization.AtPrime (p.comap (algebraMap R (Localization M))) ≃ₐ[R] Localization.AtPrime p :=
  IsLocalization.algEquiv (p.comap (algebraMap R (Localization M))).primeCompl _ _


/-- Given submonoids `M ≤ N` of `R`, this is the canonical algebra structure
of `M⁻¹S` acting on `N⁻¹S`. -/
noncomputable def localizationAlgebraOfSubmonoidLe (M N : Submonoid R) (h : M ≤ N)
    [IsLocalization M S] [IsLocalization N T] : Algebra S T :=
  (@IsLocalization.lift R _ M S _ _ T _ _ (algebraMap R T)
    (fun y => map_units T ⟨↑y, h y.prop⟩)).toAlgebra


/-- If `M ≤ N` are submonoids of `R`, then the natural map `M⁻¹S →+* N⁻¹S` commutes with the
localization maps -/
theorem localization_isScalarTower_of_submonoid_le (M N : Submonoid R) (h : M ≤ N)
    [IsLocalization M S] [IsLocalization N T] :
    @IsScalarTower R S T _ (localizationAlgebraOfSubmonoidLe S T M N h).toSMul _ :=
  letI := localizationAlgebraOfSubmonoidLe S T M N h
  IsScalarTower.of_algebraMap_eq' (IsLocalization.lift_comp _).symm


noncomputable instance (x : Ideal R) [H : x.IsPrime] [IsDomain R] :
    Algebra (Localization.AtPrime x) (Localization (nonZeroDivisors R)) :=
  localizationAlgebraOfSubmonoidLe _ _ x.primeCompl (nonZeroDivisors R)
    (by
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁴ : CommSemiring S
        inst✝³ : Algebra R S
        N : Submonoid S
        T : Type u_3
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R T
        x : Ideal R
        H : x.IsPrime
        inst✝ : IsDomain R
        ⊢ LE.le x.primeCompl (nonZeroDivisors R)
      -/
      intro a ha
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁴ : CommSemiring S
        inst✝³ : Algebra R S
        N : Submonoid S
        T : Type u_3
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R T
        x : Ideal R
        H : x.IsPrime
        inst✝ : IsDomain R
        a : R
        ha : Membership.mem x.primeCompl a
        ⊢ Membership.mem (nonZeroDivisors R) a
      -/
      rw [mem_nonZeroDivisors_iff_ne_zero]
      /-
        R : Type u_1
        inst✝⁵ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁴ : CommSemiring S
        inst✝³ : Algebra R S
        N : Submonoid S
        T : Type u_3
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R T
        x : Ideal R
        H : x.IsPrime
        inst✝ : IsDomain R
        a : R
        ha : Membership.mem x.primeCompl a
        ⊢ Ne a 0
      -/
      exact fun h => ha (h.symm ▸ x.zero_mem))
      /-
        🎉 no goals
      -/


instance {R : Type*} [CommRing R] [IsDomain R] (p : Ideal R) [p.IsPrime] :
    IsScalarTower R (Localization.AtPrime p) (FractionRing R) :=
  localization_isScalarTower_of_submonoid_le (Localization.AtPrime p) (FractionRing R)
    p.primeCompl (nonZeroDivisors R) p.primeCompl_le_nonZeroDivisors


/-- If `M ≤ N` are submonoids of `R`, then `N⁻¹S` is also the localization of `M⁻¹S` at `N`. -/
theorem isLocalization_of_submonoid_le (M N : Submonoid R) (h : M ≤ N) [IsLocalization M S]
    [IsLocalization N T] [Algebra S T] [IsScalarTower R S T] :
    IsLocalization (N.map (algebraMap R S)) T :=
  { map_units' := by
      /-
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.map (algebraMap R S) N) x) …
      -/
      rintro ⟨_, ⟨y, hy, rfl⟩⟩
      /-
        case mk.intro.intro
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        y : R
        hy : Membership.mem (↑N) y
        ⊢ IsUnit ((algebraMap S T) ↑⟨(algebraMap R S) y, ⋯⟩)
      -/
      convert IsLocalization.map_units T ⟨y, hy⟩
      /-
        case h.e'_3
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        y : R
        hy : Membership.mem (↑N) y
        ⊢ Eq ((algebraMap S T) ↑⟨(algebraMap R S) y, ⋯⟩) ((algebraMap R T) ↑⟨y, hy⟩)
      -/
      exact (IsScalarTower.algebraMap_apply _ _ _ _).symm
      /-
        🎉 no goals
      -/
    surj' := fun y => by
      /-
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        y : T
        ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap S T) ↑x.2)) ((algebraMap S T) x …
      -/
      obtain ⟨⟨x, s⟩, e⟩ := IsLocalization.surj N y
      /-
        case intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        y : T
        x : R
        s : Subtype fun x => Membership.mem N x
        e : Eq (HMul.hMul y ((algebraMap R T) ↑{ fst := x, snd := s }.2)) ((algebraMap …
        ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap S T) ↑x.2)) ((algebraMap S T) x …
      -/
      refine ⟨⟨algebraMap R S x, _, _, s.prop, rfl⟩, ?_⟩
      /-
        case intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        y : T
        x : R
        s : Subtype fun x => Membership.mem N x
        e : Eq (HMul.hMul y ((algebraMap R T) ↑{ fst := x, snd := s }.2)) ((algebraMap …
        ⊢ Eq (HMul.hMul y ((algebraMap S T) ↑{ fst := (algebraMap R S) x, snd := ⟨(alg …
      -/
      simpa [← IsScalarTower.algebraMap_apply] using e
      /-
        🎉 no goals
      -/
    exists_of_eq := fun {x₁ x₂} => by
      /-
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        ⊢ Eq ((algebraMap S T) x₁) ((algebraMap S T) x₂) → Exists fun c => Eq (HMul.hM …
      -/
      obtain ⟨⟨y₁, s₁⟩, e₁⟩ := IsLocalization.surj M x₁
      /-
        case intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        y₁ : R
        s₁ : Subtype fun x => Membership.mem M x
        e₁ : Eq (HMul.hMul x₁ ((algebraMap R S) ↑{ fst := y₁, snd := s₁ }.2)) ((algebr …
        ⊢ Eq ((algebraMap S T) x₁) ((algebraMap S T) x₂) → Exists fun c => Eq (HMul.hM …
      -/
      obtain ⟨⟨y₂, s₂⟩, e₂⟩ := IsLocalization.surj M x₂
      /-
        case intro.mk.intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        y₁ : R
        s₁ : Subtype fun x => Membership.mem M x
        e₁ : Eq (HMul.hMul x₁ ((algebraMap R S) ↑{ fst := y₁, snd := s₁ }.2)) ((algebr …
        y₂ : R
        s₂ : Subtype fun x => Membership.mem M x
        e₂ : Eq (HMul.hMul x₂ ((algebraMap R S) ↑{ fst := y₂, snd := s₂ }.2)) ((algebr …
        ⊢ Eq ((algebraMap S T) x₁) ((algebraMap S T) x₂) → Exists fun c => Eq (HMul.hM …
      -/
      refine (Set.exists_image_iff (algebraMap R S) N fun c => c * x₁ = c * x₂).mpr.comp ?_
      /-
        case intro.mk.intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        y₁ : R
        s₁ : Subtype fun x => Membership.mem M x
        e₁ : Eq (HMul.hMul x₁ ((algebraMap R S) ↑{ fst := y₁, snd := s₁ }.2)) ((algebr …
        y₂ : R
        s₂ : Subtype fun x => Membership.mem M x
        e₂ : Eq (HMul.hMul x₂ ((algebraMap R S) ↑{ fst := y₂, snd := s₂ }.2)) ((algebr …
        ⊢ Eq ((algebraMap S T) x₁) ((algebraMap S T) x₂) → Exists fun a => Eq (HMul.hM …
      -/
      dsimp only at e₁ e₂ ⊢
      suffices algebraMap R T (y₁ * s₂) = algebraMap R T (y₂ * s₁) →
          ∃ a : N, algebraMap R S (a * (y₁ * s₂)) = algebraMap R S (a * (y₂ * s₁)) by
        have h₁ := @IsUnit.mul_left_inj T _ _ (algebraMap S T x₁) (algebraMap S T x₂)
          (IsLocalization.map_units T ⟨(s₁ : R), h s₁.prop⟩)
        have h₂ := @IsUnit.mul_left_inj T _ _ ((algebraMap S T x₁) * (algebraMap R T s₁))
          ((algebraMap S T x₂) * (algebraMap R T s₁))
          (IsLocalization.map_units T ⟨(s₂ : R), h s₂.prop⟩)
        simp only [IsScalarTower.algebraMap_apply R S T, Subtype.coe_mk] at h₁ h₂
        simp only [IsScalarTower.algebraMap_apply R S T, map_mul, ← e₁, ← e₂, ← mul_assoc,
          mul_right_comm _ (algebraMap R S s₂),
          mul_right_comm _ (algebraMap S T (algebraMap R S s₂)),
          (IsLocalization.map_units S s₁).mul_left_inj,
          (IsLocalization.map_units S s₂).mul_left_inj] at this
        rw [h₂, h₁] at this
        simpa only [mul_comm] using this
      /-
        case intro.mk.intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        y₁ : R
        s₁ : Subtype fun x => Membership.mem M x
        e₁ : Eq (HMul.hMul x₁ ((algebraMap R S) ↑s₁)) ((algebraMap R S) y₁)
        y₂ : R
        s₂ : Subtype fun x => Membership.mem M x
        e₂ : Eq (HMul.hMul x₂ ((algebraMap R S) ↑s₂)) ((algebraMap R S) y₂)
        ⊢ Eq ((algebraMap R T) (HMul.hMul y₁ ↑s₂)) ((algebraMap R T) (HMul.hMul y₂ ↑s₁ …
      -/
      simp_rw [IsLocalization.eq_iff_exists N T, IsLocalization.eq_iff_exists M S]
      /-
        case intro.mk.intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        y₁ : R
        s₁ : Subtype fun x => Membership.mem M x
        e₁ : Eq (HMul.hMul x₁ ((algebraMap R S) ↑s₁)) ((algebraMap R S) y₁)
        y₂ : R
        s₂ : Subtype fun x => Membership.mem M x
        e₂ : Eq (HMul.hMul x₂ ((algebraMap R S) ↑s₂)) ((algebraMap R S) y₂)
        ⊢ (Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul y₁ ↑s₂)) (HMul.hMul (↑c) (HMu …
      -/
      intro ⟨a, e⟩
      /-
        case intro.mk.intro.mk
        R : Type u_1
        inst✝⁸ : CommSemiring R
        S : Type u_2
        inst✝⁷ : CommSemiring S
        inst✝⁶ : Algebra R S
        T : Type u_3
        inst✝⁵ : CommSemiring T
        inst✝⁴ : Algebra R T
        M N : Submonoid R
        h : LE.le M N
        inst✝³ : IsLocalization M S
        inst✝² : IsLocalization N T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        x₁ x₂ : S
        y₁ : R
        s₁ : Subtype fun x => Membership.mem M x
        e₁ : Eq (HMul.hMul x₁ ((algebraMap R S) ↑s₁)) ((algebraMap R S) y₁)
        y₂ : R
        s₂ : Subtype fun x => Membership.mem M x
        e₂ : Eq (HMul.hMul x₂ ((algebraMap R S) ↑s₂)) ((algebraMap R S) y₂)
        a : Subtype fun x => Membership.mem N x
        e : Eq (HMul.hMul (↑a) (HMul.hMul y₁ ↑s₂)) (HMul.hMul (↑a) (HMul.hMul y₂ ↑s₁))
        ⊢ Exists fun a => Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (↑a) (HMul.hMu …
      -/
      exact ⟨a, 1, by convert e using 1 <;> simp⟩ }
      /-
        🎉 no goals
      -/


/-- If `M ≤ N` are submonoids of `R` such that `∀ x : N, ∃ m : R, m * x ∈ M`, then the
localization at `N` is equal to the localizaton of `M`. -/
theorem isLocalization_of_is_exists_mul_mem (M N : Submonoid R) [IsLocalization M S] (h : M ≤ N)
    (h' : ∀ x : N, ∃ m : R, m * x ∈ M) : IsLocalization N S :=
  { map_units' := fun y => by
      /-
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        y : Subtype fun x => Membership.mem N x
        ⊢ IsUnit ((algebraMap R S) ↑y)
      -/
      obtain ⟨m, hm⟩ := h' y
      /-
        case intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        y : Subtype fun x => Membership.mem N x
        m : R
        hm : Membership.mem M (HMul.hMul m ↑y)
        ⊢ IsUnit ((algebraMap R S) ↑y)
      -/
      have := IsLocalization.map_units S ⟨_, hm⟩
      /-
        case intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        y : Subtype fun x => Membership.mem N x
        m : R
        hm : Membership.mem M (HMul.hMul m ↑y)
        this : IsUnit ((algebraMap R S) ↑⟨HMul.hMul m ↑y, hm⟩)
        ⊢ IsUnit ((algebraMap R S) ↑y)
      -/
      rw [map_mul] at this
      /-
        case intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        y : Subtype fun x => Membership.mem N x
        m : R
        hm : Membership.mem M (HMul.hMul m ↑y)
        this : IsUnit (HMul.hMul ((algebraMap R S) m) ((algebraMap R S) ↑y))
        ⊢ IsUnit ((algebraMap R S) ↑y)
      -/
      exact (IsUnit.mul_iff.mp this).2
      /-
        🎉 no goals
      -/
    surj' := fun z => by
      /-
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        z : S
        ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap R S) ↑x.2)) ((algebraMap R S) x …
      -/
      obtain ⟨⟨y, s⟩, e⟩ := IsLocalization.surj M z
      /-
        case intro.mk
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        z : S
        y : R
        s : Subtype fun x => Membership.mem M x
        e : Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
        ⊢ Exists fun x => Eq (HMul.hMul z ((algebraMap R S) ↑x.2)) ((algebraMap R S) x …
      -/
      exact ⟨⟨y, _, h s.prop⟩, e⟩
      /-
        🎉 no goals
      -/
    exists_of_eq := fun {_ _} => by
      /-
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        x✝¹ x✝ : R
        ⊢ Eq ((algebraMap R S) x✝¹) ((algebraMap R S) x✝) → Exists fun c => Eq (HMul.h …
      -/
      rw [IsLocalization.eq_iff_exists M]
      /-
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        M N : Submonoid R
        inst✝ : IsLocalization M S
        h : LE.le M N
        h' : ∀ (x : Subtype fun x => Membership.mem N x), Exists fun m => Membership.m …
        x✝¹ x✝ : R
        ⊢ (Exists fun c => Eq (HMul.hMul (↑c) x✝¹) (HMul.hMul (↑c) x✝)) → Exists fun c …
      -/
      exact fun ⟨x, hx⟩ => ⟨⟨_, h x.prop⟩, hx⟩ }
      /-
        🎉 no goals
      -/


theorem mk'_eq_algebraMap_mk'_of_submonoid_le {M N : Submonoid R} (h : M ≤ N) [IsLocalization M S]
    [IsLocalization N T] [Algebra S T] [IsScalarTower R S T] (x : R) (y : {a : R // a ∈ M}) :
    mk' T x ⟨y.1, h y.2⟩ = algebraMap S T (mk' S x y) :=
                            /-
                              R : Type u_1
                              inst✝⁸ : CommSemiring R
                              S : Type u_2
                              inst✝⁷ : CommSemiring S
                              inst✝⁶ : Algebra R S
                              T : Type u_3
                              inst✝⁵ : CommSemiring T
                              inst✝⁴ : Algebra R T
                              M N : Submonoid R
                              h : LE.le M N
                              inst✝³ : IsLocalization M S
                              inst✝² : IsLocalization N T
                              inst✝¹ : Algebra S T
                              inst✝ : IsScalarTower R S T
                              x : R
                              y : Subtype fun a => Membership.mem M a
                              ⊢ Eq ((algebraMap R T) x) (HMul.hMul ((algebraMap S T) (IsLocalization.mk' S x …
                            -/
  mk'_eq_iff_eq_mul.mpr (by simp only [IsScalarTower.algebraMap_apply R S T, ← map_mul, mk'_spec])
                            /-
                              🎉 no goals
                            -/


theorem isFractionRing_of_isLocalization (S T : Type*) [CommRing S] [CommRing T] [Algebra R S]
    [Algebra R T] [Algebra S T] [IsScalarTower R S T] [IsLocalization M S] [IsFractionRing R T]
    (hM : M ≤ nonZeroDivisors R) : IsFractionRing S T := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Submonoid R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    hM : LE.le M (nonZeroDivisors R)
    ⊢ IsFractionRing S T
  -/
  have := isLocalization_of_submonoid_le S T M (nonZeroDivisors R) hM
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Submonoid R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    hM : LE.le M (nonZeroDivisors R)
    this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
    ⊢ IsFractionRing S T
  -/
  refine @isLocalization_of_is_exists_mul_mem _ _ _ _ _ _ _ this ?_ ?_
    /-
      case refine_1
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      ⊢ LE.le (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) (nonZeroDivisors S)
    -/
  · exact map_nonZeroDivisors_le M S
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      ⊢ ∀ (x : Subtype fun x => Membership.mem (nonZeroDivisors S) x), Exists fun m  …
    -/
  · rintro ⟨x, hx⟩
    /-
      case refine_2.mk
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      ⊢ Exists fun m => Membership.mem (Submonoid.map (algebraMap R S) (nonZeroDivis …
    -/
    obtain ⟨⟨y, s⟩, e⟩ := IsLocalization.surj M x
    /-
      case refine_2.mk.intro.mk
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      ⊢ Exists fun m => Membership.mem (Submonoid.map (algebraMap R S) (nonZeroDivis …
    -/
    use algebraMap R S s
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      ⊢ Membership.mem (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) (HMul.hM …
    -/
    rw [mul_comm, Subtype.coe_mk, e]
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      ⊢ Membership.mem (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) ((algebr …
    -/
    refine Set.mem_image_of_mem (algebraMap R S) ?_
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      ⊢ Membership.mem ↑(nonZeroDivisors R) { fst := y, snd := s }.1
    -/
    intro z hz
    /-
      case h
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      z : R
      hz : Eq (HMul.hMul z { fst := y, snd := s }.1) 0
      ⊢ Eq z 0
    -/
    apply IsLocalization.injective S hM
    /-
      case h.a
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      z : R
      hz : Eq (HMul.hMul z { fst := y, snd := s }.1) 0
      ⊢ Eq ((algebraMap R S) z) ((algebraMap R S) 0)
    -/
    rw [map_zero]
    /-
      case h.a
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Submonoid R
      S : Type u_2
      T : Type u_3
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : IsLocalization M S
      inst✝ : IsFractionRing R T
      hM : LE.le M (nonZeroDivisors R)
      this : IsLocalization (Submonoid.map (algebraMap R S) (nonZeroDivisors R)) T
      x : S
      hx : Membership.mem (nonZeroDivisors S) x
      y : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := s }.2)) ((algebraMap …
      z : R
      hz : Eq (HMul.hMul z { fst := y, snd := s }.1) 0
      ⊢ Eq ((algebraMap R S) z) 0
    -/
    apply hx
    rw [← (map_units S s).mul_left_inj, mul_assoc, e, ← map_mul, hz, map_zero,
      zero_mul]


theorem isFractionRing_of_isDomain_of_isLocalization [IsDomain R] (S T : Type*) [CommRing S]
    [CommRing T] [Algebra R S] [Algebra R T] [Algebra S T] [IsScalarTower R S T]
    [IsLocalization M S] [IsFractionRing R T] : IsFractionRing S T := by
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    ⊢ IsFractionRing S T
  -/
  haveI := IsFractionRing.nontrivial R T
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this : Nontrivial T
    ⊢ IsFractionRing S T
  -/
  haveI := (algebraMap S T).domain_nontrivial
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    ⊢ IsFractionRing S T
  -/
  apply isFractionRing_of_isLocalization M S T
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    ⊢ LE.le M (nonZeroDivisors R)
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    x : R
    hx : Membership.mem M x
    ⊢ Membership.mem (nonZeroDivisors R) x
  -/
  rw [mem_nonZeroDivisors_iff_ne_zero]
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    x : R
    hx : Membership.mem M x
    ⊢ Ne x 0
  -/
  intro hx'
  /-
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    x : R
    hx : Membership.mem M x
    hx' : Eq x 0
    ⊢ False
  -/
  apply @zero_ne_one S
  /-
    case a
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    x : R
    hx : Membership.mem M x
    hx' : Eq x 0
    ⊢ Eq 0 1
  -/
  rw [← (algebraMap R S).map_one, ← @mk'_one R _ M, @comm _ Eq, mk'_eq_zero_iff]
  /-
    case a
    R : Type u_1
    inst✝⁹ : CommRing R
    M : Submonoid R
    inst✝⁸ : IsDomain R
    S : Type u_2
    T : Type u_3
    inst✝⁷ : CommRing S
    inst✝⁶ : CommRing T
    inst✝⁵ : Algebra R S
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    inst✝¹ : IsLocalization M S
    inst✝ : IsFractionRing R T
    this✝ : Nontrivial T
    this : Nontrivial S
    x : R
    hx : Membership.mem M x
    hx' : Eq x 0
    ⊢ Exists fun m => Eq (HMul.hMul (↑m) 1) 0
  -/
  exact ⟨⟨x, hx⟩, by simp [hx']⟩
  /-
    🎉 no goals
  -/


instance {R : Type*} [CommRing R] [IsDomain R] (p : Ideal R) [p.IsPrime] :
    IsFractionRing (Localization.AtPrime p) (FractionRing R) :=
  IsFractionRing.isFractionRing_of_isDomain_of_isLocalization p.primeCompl
    (Localization.AtPrime p) (FractionRing R)


