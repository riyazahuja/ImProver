/--
An `R`-algebra is essentially of finite type if
it is the localization of an algebra of finite type.
See `essFiniteType_iff_exists_subalgebra`.
-/
class EssFiniteType : Prop where
  cond : ∃ s : Finset S,
    IsLocalization ((IsUnit.submonoid S).comap (algebraMap (adjoin R (s : Set S)) S)) S


/-- Let `S` be an `R`-algebra essentially of finite type, this is a choice of a finset `s ⊆ S`
such that `S` is the localization of `R[s]`. -/
noncomputable
def EssFiniteType.finset [h : EssFiniteType R S] : Finset S := h.cond.choose


/-- A choice of a subalgebra of finite type in an essentially of finite type algebra, such that
its localization is the whole ring. -/
noncomputable
abbrev EssFiniteType.subalgebra [EssFiniteType R S] : Subalgebra R S :=
  Algebra.adjoin R (finset R S : Set S)


lemma EssFiniteType.adjoin_mem_finset [EssFiniteType R S] :
    adjoin R { x : subalgebra R S | x.1 ∈ finset R S } = ⊤ := adjoin_adjoin_coe_preimage


instance [EssFiniteType R S] : Algebra.FiniteType R (EssFiniteType.subalgebra R S) := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.EssFiniteType R S
    ⊢ Algebra.FiniteType R (Subtype fun x => Membership.mem (Algebra.EssFiniteType …
  -/
  constructor
  /-
    case out
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.EssFiniteType R S
    ⊢ Top.top.FG
  -/
  rw [Subalgebra.fg_top, EssFiniteType.subalgebra]
  /-
    case out
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.EssFiniteType R S
    ⊢ (Algebra.adjoin R ↑(Algebra.EssFiniteType.finset R S)).FG
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


/-- A submonoid of `EssFiniteType.subalgebra R S`, whose localization is the whole algebra `S`. -/
noncomputable
def EssFiniteType.submonoid [EssFiniteType R S] : Submonoid (EssFiniteType.subalgebra R S) :=
  ((IsUnit.submonoid S).comap (algebraMap (EssFiniteType.subalgebra R S) S))


instance EssFiniteType.isLocalization [h : EssFiniteType R S] :
    IsLocalization (EssFiniteType.submonoid R S) S :=
  h.cond.choose_spec


lemma essFiniteType_cond_iff (σ : Finset S) :
    IsLocalization ((IsUnit.submonoid S).comap (algebraMap (adjoin R (σ : Set S)) S)) S ↔
    (∀ s : S, ∃ t ∈ Algebra.adjoin R (σ : Set S),
      IsUnit t ∧ s * t ∈ Algebra.adjoin R (σ : Set S)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    σ : Finset S
    ⊢ Iff (IsLocalization (Submonoid.comap (algebraMap (Subtype fun x => Membershi …
  -/
  constructor <;> intro hσ
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      σ : Finset S
      hσ : IsLocalization (Submonoid.comap (algebraMap (Subtype fun x => Membership. …
      ⊢ ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) (And …
    -/
  · intro s
    /-
      case mp
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      σ : Finset S
      hσ : IsLocalization (Submonoid.comap (algebraMap (Subtype fun x => Membership. …
      s : S
      ⊢ Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) (And (IsUnit t) …
    -/
    obtain ⟨⟨⟨x, hx⟩, ⟨t, ht⟩, ht'⟩, h⟩ := hσ.2 s
    /-
      case mp.intro.mk.mk.mk.mk
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      σ : Finset S
      hσ : IsLocalization (Submonoid.comap (algebraMap (Subtype fun x => Membership. …
      s x : S
      hx : Membership.mem (Algebra.adjoin R ↑σ) x
      t : S
      ht : Membership.mem (Algebra.adjoin R ↑σ) t
      ht' : Membership.mem (Submonoid.comap (algebraMap (Subtype fun x => Membership …
      h : Eq (HMul.hMul s ((algebraMap (Subtype fun x => Membership.mem (Algebra.adj …
      ⊢ Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) (And (IsUnit t) …
    -/
    exact ⟨t, ht, ht', h ▸ hx⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      σ : Finset S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
      ⊢ IsLocalization (Submonoid.comap (algebraMap (Subtype fun x => Membership.mem …
    -/
  · constructor
      /-
        case mpr.map_units'
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        σ : Finset S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
        ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.comap (algebraMap (Subtype …
      -/
    · exact fun y ↦ y.prop
      /-
        🎉 no goals
      -/
      /-
        case mpr.surj'
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        σ : Finset S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
        ⊢ ∀ (z : S), Exists fun x => Eq (HMul.hMul z ((algebraMap (Subtype fun x => Me …
      -/
    · intro s
      /-
        case mpr.surj'
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        σ : Finset S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
        s : S
        ⊢ Exists fun x => Eq (HMul.hMul s ((algebraMap (Subtype fun x => Membership.me …
      -/
      obtain ⟨t, ht, ht', h⟩ := hσ s
      /-
        case mpr.surj'.intro.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        σ : Finset S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
        s t : S
        ht : Membership.mem (Algebra.adjoin R ↑σ) t
        ht' : IsUnit t
        h : Membership.mem (Algebra.adjoin R ↑σ) (HMul.hMul s t)
        ⊢ Exists fun x => Eq (HMul.hMul s ((algebraMap (Subtype fun x => Membership.me …
      -/
      exact ⟨⟨⟨_, h⟩, ⟨t, ht⟩, ht'⟩, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.exists_of_eq
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        σ : Finset S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
        ⊢ ∀ {x y : Subtype fun x => Membership.mem (Algebra.adjoin R ↑σ) x}, Eq ((alge …
      -/
    · intros x y e
      /-
        case mpr.exists_of_eq
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        σ : Finset S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjoin R ↑σ) t) ( …
        x y : Subtype fun x => Membership.mem (Algebra.adjoin R ↑σ) x
        e : Eq ((algebraMap (Subtype fun x => Membership.mem (Algebra.adjoin R ↑σ) x)  …
        ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      -/
      exact ⟨1, by simpa using Subtype.ext e⟩
      /-
        🎉 no goals
      -/


lemma essFiniteType_iff :
    EssFiniteType R S ↔ ∃ (σ : Finset S),
      (∀ s : S, ∃ t ∈ Algebra.adjoin R (σ : Set S),
        IsUnit t ∧ s * t ∈ Algebra.adjoin R (σ : Set S)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.EssFiniteType R S) (Exists fun σ => ∀ (s : S), Exists fun t =>  …
  -/
  simp_rw [← essFiniteType_cond_iff]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.EssFiniteType R S) (Exists fun σ => IsLocalization (Submonoid.c …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> exact fun ⟨a, b⟩ ↦ ⟨a, b⟩
                  /-
                    🎉 no goals
                  -/


instance EssFiniteType.of_finiteType [FiniteType R S] : EssFiniteType R S := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.FiniteType R S
    ⊢ Algebra.EssFiniteType R S
  -/
  obtain ⟨s, hs⟩ := ‹FiniteType R S›
  /-
    case mk.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.FiniteType R S
    s : Finset S
    hs : Eq (Algebra.adjoin R ↑s) Top.top
    ⊢ Algebra.EssFiniteType R S
  -/
  rw [essFiniteType_iff]
  /-
    case mk.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.FiniteType R S
    s : Finset S
    hs : Eq (Algebra.adjoin R ↑s) Top.top
    ⊢ Exists fun σ => ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjo …
  -/
  exact ⟨s, fun _ ↦ by simpa only [hs, mem_top, and_true, true_and] using ⟨1, isUnit_one⟩⟩
  /-
    🎉 no goals
  -/


variable {R} in
lemma EssFiniteType.of_isLocalization (M : Submonoid R) [IsLocalization M S] :
    EssFiniteType R S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    ⊢ Algebra.EssFiniteType R S
  -/
  rw [essFiniteType_iff]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    ⊢ Exists fun σ => ∀ (s : S), Exists fun t => And (Membership.mem (Algebra.adjo …
  -/
  use ∅
  simp only [Finset.coe_empty, Algebra.adjoin_empty, exists_and_left, Algebra.mem_bot,
    Set.mem_range, exists_exists_eq_and]
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    ⊢ ∀ (s : S), Exists fun a => And (IsUnit ((algebraMap R S) a)) (Exists fun y = …
  -/
  intro s
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    s : S
    ⊢ Exists fun a => And (IsUnit ((algebraMap R S) a)) (Exists fun y => Eq ((alge …
  -/
  obtain ⟨⟨x, t⟩, e⟩ := IsLocalization.surj M s
  /-
    case h.intro.mk
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    M : Submonoid R
    inst✝ : IsLocalization M S
    s : S
    x : R
    t : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul s ((algebraMap R S) ↑{ fst := x, snd := t }.2)) ((algebraMap …
    ⊢ Exists fun a => And (IsUnit ((algebraMap R S) a)) (Exists fun y => Eq ((alge …
  -/
  exact ⟨_, IsLocalization.map_units S t, x, e.symm⟩
  /-
    🎉 no goals
  -/


lemma EssFiniteType.of_id : EssFiniteType R R := inferInstance


lemma EssFiniteType.aux (σ : Subalgebra R S)
    (hσ : ∀ s : S, ∃ t ∈ σ, IsUnit t ∧ s * t ∈ σ)
    (τ : Set T) (t : T) (ht : t ∈ Algebra.adjoin S τ) :
    ∃ s ∈ σ, IsUnit s ∧ s • t ∈ σ.map (IsScalarTower.toAlgHom R S T) ⊔ Algebra.adjoin R τ := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R S
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    σ : Subalgebra R S
    hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
    τ : Set T
    t : T
    ht : Membership.mem (Algebra.adjoin S τ) t
    ⊢ Exists fun s => And (Membership.mem σ s) (And (IsUnit s) (Membership.mem (Ma …
  -/
  refine Algebra.adjoin_induction ?_ ?_ ?_ ?_ ht
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      ⊢ ∀ (x : T), Membership.mem τ x → Exists fun s => And (Membership.mem σ s) (An …
    -/
  · intro t ht
    exact ⟨1, Subalgebra.one_mem _, isUnit_one,
      (one_smul S t).symm ▸ Algebra.mem_sup_right (Algebra.subset_adjoin ht)⟩
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      ⊢ ∀ (r : S), Exists fun s => And (Membership.mem σ s) (And (IsUnit s) (Members …
    -/
  · intro s
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      s : S
      ⊢ Exists fun s_1 => And (Membership.mem σ s_1) (And (IsUnit s_1) (Membership.m …
    -/
    obtain ⟨s', hs₁, hs₂, hs₃⟩ := hσ s
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      s s' : S
      hs₁ : Membership.mem σ s'
      hs₂ : IsUnit s'
      hs₃ : Membership.mem σ (HMul.hMul s s')
      ⊢ Exists fun s_1 => And (Membership.mem σ s_1) (And (IsUnit s_1) (Membership.m …
    -/
    refine ⟨_, hs₁, hs₂, Algebra.mem_sup_left ?_⟩
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      s s' : S
      hs₁ : Membership.mem σ s'
      hs₂ : IsUnit s'
      hs₃ : Membership.mem σ (HMul.hMul s s')
      ⊢ Membership.mem (Subalgebra.map (IsScalarTower.toAlgHom R S T) σ) (HSMul.hSMu …
    -/
    rw [Algebra.smul_def, ← map_mul, mul_comm]
    /-
      case refine_2.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      s s' : S
      hs₁ : Membership.mem σ s'
      hs₂ : IsUnit s'
      hs₃ : Membership.mem σ (HMul.hMul s s')
      ⊢ Membership.mem (Subalgebra.map (IsScalarTower.toAlgHom R S T) σ) ((algebraMa …
    -/
    exact ⟨_, hs₃, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      ⊢ ∀ (x y : T), Membership.mem (Algebra.adjoin S τ) x → Membership.mem (Algebra …
    -/
  · rintro x y - - ⟨sx, hsx, hsx', hsx''⟩ ⟨sy, hsy, hsy', hsy''⟩
    /-
      case refine_3.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      x y : T
      sx : S
      hsx : Membership.mem σ sx
      hsx' : IsUnit sx
      hsx'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      sy : S
      hsy : Membership.mem σ sy
      hsy' : IsUnit sy
      hsy'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      ⊢ Exists fun s => And (Membership.mem σ s) (And (IsUnit s) (Membership.mem (Ma …
    -/
    refine ⟨_, σ.mul_mem hsx hsy, hsx'.mul hsy', ?_⟩
    rw [smul_add, mul_smul, mul_smul, Algebra.smul_def sx (sy • y), smul_comm,
      Algebra.smul_def sy (sx • x)]
    /-
      case refine_3.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      x y : T
      sx : S
      hsx : Membership.mem σ sx
      hsx' : IsUnit sx
      hsx'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      sy : S
      hsy : Membership.mem σ sy
      hsy' : IsUnit sy
      hsy'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      ⊢ Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) σ) (A …
    -/
    apply add_mem (mul_mem _ hsx'') (mul_mem _ hsy'') <;>
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : CommRing T
        inst✝³ : Algebra R S
        inst✝² : Algebra R T
        inst✝¹ : Algebra S T
        inst✝ : IsScalarTower R S T
        σ : Subalgebra R S
        hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
        τ : Set T
        t : T
        ht : Membership.mem (Algebra.adjoin S τ) t
        x y : T
        sx : S
        hsx : Membership.mem σ sx
        hsx' : IsUnit sx
        hsx'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
        sy : S
        hsy : Membership.mem σ sy
        hsy' : IsUnit sy
        hsy'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
        ⊢ Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) σ) (A …
      -/
      /-
        🎉 no goals
      -/
      exact Algebra.mem_sup_left ⟨_, ‹_›, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_4
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      ⊢ ∀ (x y : T), Membership.mem (Algebra.adjoin S τ) x → Membership.mem (Algebra …
    -/
  · rintro x y - - ⟨sx, hsx, hsx', hsx''⟩ ⟨sy, hsy, hsy', hsy''⟩
    /-
      case refine_4.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      x y : T
      sx : S
      hsx : Membership.mem σ sx
      hsx' : IsUnit sx
      hsx'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      sy : S
      hsy : Membership.mem σ sy
      hsy' : IsUnit sy
      hsy'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      ⊢ Exists fun s => And (Membership.mem σ s) (And (IsUnit s) (Membership.mem (Ma …
    -/
    refine ⟨_, σ.mul_mem hsx hsy, hsx'.mul hsy', ?_⟩
    /-
      case refine_4.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      x y : T
      sx : S
      hsx : Membership.mem σ sx
      hsx' : IsUnit sx
      hsx'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      sy : S
      hsy : Membership.mem σ sy
      hsy' : IsUnit sy
      hsy'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      ⊢ Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) σ) (A …
    -/
    rw [mul_smul, ← smul_eq_mul, smul_comm sy x, ← smul_assoc, smul_eq_mul]
    /-
      case refine_4.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Subalgebra R S
      hσ : ∀ (s : S), Exists fun t => And (Membership.mem σ t) (And (IsUnit t) (Memb …
      τ : Set T
      t : T
      ht : Membership.mem (Algebra.adjoin S τ) t
      x y : T
      sx : S
      hsx : Membership.mem σ sx
      hsx' : IsUnit sx
      hsx'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      sy : S
      hsy : Membership.mem σ sy
      hsy' : IsUnit sy
      hsy'' : Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) …
      ⊢ Membership.mem (Max.max (Subalgebra.map (IsScalarTower.toAlgHom R S T) σ) (A …
    -/
    exact mul_mem hsx'' hsy''
    /-
      🎉 no goals
    -/


lemma EssFiniteType.comp [h₁ : EssFiniteType R S] [h₂ : EssFiniteType S T] :
    EssFiniteType R T := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R S
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    h₁ : Algebra.EssFiniteType R S
    h₂ : Algebra.EssFiniteType S T
    ⊢ Algebra.EssFiniteType R T
  -/
  rw [essFiniteType_iff] at h₁ h₂ ⊢
  classical
  obtain ⟨s, hs⟩ := h₁
  obtain ⟨t, ht⟩ := h₂
  use s.image (IsScalarTower.toAlgHom R S T) ∪ t
  simp only [Finset.coe_union, Finset.coe_image, Algebra.adjoin_union, Algebra.adjoin_image]
  intro x
  obtain ⟨y, hy₁, hy₂, hy₃⟩ := ht x
  obtain ⟨t₁, h₁, h₂, h₃⟩ := EssFiniteType.aux _ _ _ _ hs _ y hy₁
  obtain ⟨t₂, h₄, h₅, h₆⟩ := EssFiniteType.aux _ _ _ _ hs _ _ hy₃
  refine ⟨t₂ • t₁ • y, ?_, ?_, ?_⟩
  · rw [Algebra.smul_def]
    exact mul_mem (Algebra.mem_sup_left ⟨_, h₄, rfl⟩) h₃
  · rw [Algebra.smul_def, Algebra.smul_def]
    exact (h₅.map _).mul ((h₂.map _).mul hy₂)
  · rw [← mul_smul, mul_comm, smul_mul_assoc, mul_comm, mul_comm y, mul_smul, Algebra.smul_def]
    exact mul_mem (Algebra.mem_sup_left ⟨_, h₁, rfl⟩) h₆


open EssFiniteType in
lemma essFiniteType_iff_exists_subalgebra : EssFiniteType R S ↔
    ∃ (S₀ : Subalgebra R S) (M : Submonoid S₀), FiniteType R S₀ ∧ IsLocalization M S := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Iff (Algebra.EssFiniteType R S) (Exists fun S₀ => Exists fun M => And (Algeb …
  -/
  refine ⟨fun h ↦ ⟨subalgebra R S, submonoid R S, inferInstance, inferInstance⟩, ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ (Exists fun S₀ => Exists fun M => And (Algebra.FiniteType R (Subtype fun x = …
  -/
  rintro ⟨S₀, M, _, _⟩
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S₀ : Subalgebra R S
    M : Submonoid (Subtype fun x => Membership.mem S₀ x)
    left✝ : Algebra.FiniteType R (Subtype fun x => Membership.mem S₀ x)
    right✝ : IsLocalization M S
    ⊢ Algebra.EssFiniteType R S
  -/
  letI := of_isLocalization S M
  /-
    case intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S₀ : Subalgebra R S
    M : Submonoid (Subtype fun x => Membership.mem S₀ x)
    left✝ : Algebra.FiniteType R (Subtype fun x => Membership.mem S₀ x)
    right✝ : IsLocalization M S
    this : Algebra.EssFiniteType (Subtype fun x => Membership.mem S₀ x) S := Algeb …
    ⊢ Algebra.EssFiniteType R S
  -/
  exact comp R S₀ S
  /-
    🎉 no goals
  -/


instance EssFiniteType.baseChange [h : EssFiniteType R S] : EssFiniteType T (T ⊗[R] S) := by
  classical
  rw [essFiniteType_iff] at h ⊢
  obtain ⟨σ, hσ⟩ := h
  use σ.image Algebra.TensorProduct.includeRight
  intro s
  induction' s using TensorProduct.induction_on with x y x y hx hy
  · exact ⟨1, one_mem _, isUnit_one, by simpa using zero_mem _⟩
  · obtain ⟨t, h₁, h₂, h₃⟩ := hσ y
    have H (x : S) (hx : x ∈ Algebra.adjoin R (σ : Set S)) :
        1 ⊗ₜ[R] x ∈ Algebra.adjoin T
          ((σ.image Algebra.TensorProduct.includeRight : Finset (T ⊗[R] S)) : Set (T ⊗[R] S)) := by
      have : Algebra.TensorProduct.includeRight x ∈
          (Algebra.adjoin R (σ : Set S)).map (Algebra.TensorProduct.includeRight (A := T)) :=
        Subalgebra.mem_map.mpr ⟨_, hx, rfl⟩
      rw [← Algebra.adjoin_adjoin_of_tower R]
      apply Algebra.subset_adjoin
      simpa [← Algebra.adjoin_image] using this
    refine ⟨Algebra.TensorProduct.includeRight t, H _ h₁, h₂.map _, ?_⟩
    simp only [Algebra.TensorProduct.includeRight_apply, Algebra.TensorProduct.tmul_mul_tmul,
      mul_one]
    rw [← mul_one x, ← smul_eq_mul, ← TensorProduct.smul_tmul']
    apply Subalgebra.smul_mem
    exact H _ h₃
  · obtain ⟨tx, hx₁, hx₂, hx₃⟩ := hx
    obtain ⟨ty, hy₁, hy₂, hy₃⟩ := hy
    refine ⟨_, mul_mem hx₁ hy₁, hx₂.mul hy₂, ?_⟩
    rw [add_mul, ← mul_assoc, mul_comm tx ty, ← mul_assoc]
    exact add_mem (mul_mem hx₃ hy₁) (mul_mem hy₃ hx₁)


lemma EssFiniteType.of_comp [h : EssFiniteType R T] : EssFiniteType S T := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R S
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    h : Algebra.EssFiniteType R T
    ⊢ Algebra.EssFiniteType S T
  -/
  rw [essFiniteType_iff] at h ⊢
  classical
  obtain ⟨σ, hσ⟩ := h
  use σ
  intro x
  obtain ⟨y, hy₁, hy₂, hy₃⟩ := hσ x
  simp_rw [← Algebra.adjoin_adjoin_of_tower R (S := S) (σ : Set T)]
  exact ⟨y, Algebra.subset_adjoin hy₁, hy₂, Algebra.subset_adjoin hy₃⟩


lemma EssFiniteType.comp_iff [EssFiniteType R S] :
    EssFiniteType R T ↔ EssFiniteType S T :=
  ⟨fun _ ↦ of_comp R S T, fun _ ↦ comp R S T⟩


variable {R S} in
lemma EssFiniteType.algHom_ext [EssFiniteType R S]
    (f g : S →ₐ[R] T) (H : ∀ s ∈ finset R S, f s = g s) : f = g := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.EssFiniteType R S
    f g : AlgHom R S T
    H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
    ⊢ Eq f g
  -/
  suffices f.toRingHom = g.toRingHom by ext; exact RingHom.congr_fun this _
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.EssFiniteType R S
    f g : AlgHom R S T
    H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
    ⊢ Eq f.toRingHom g.toRingHom
  -/
  apply IsLocalization.ringHom_ext (EssFiniteType.submonoid R S)
  suffices f.comp (IsScalarTower.toAlgHom R _ S) = g.comp (IsScalarTower.toAlgHom R _ S) by
    ext; exact AlgHom.congr_fun this _
  /-
    case h
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing T
    inst✝² : Algebra R S
    inst✝¹ : Algebra R T
    inst✝ : Algebra.EssFiniteType R S
    f g : AlgHom R S T
    H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
    ⊢ Eq (f.comp (IsScalarTower.toAlgHom R (Subtype fun x => Membership.mem (Algeb …
  -/
  apply AlgHom.ext_of_adjoin_eq_top (s := { x | x.1 ∈ finset R S })
    /-
      case h.h
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : Algebra R S
      inst✝¹ : Algebra R T
      inst✝ : Algebra.EssFiniteType R S
      f g : AlgHom R S T
      H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
      ⊢ Eq (Algebra.adjoin R (setOf fun x => Membership.mem (Algebra.EssFiniteType.f …
    -/
  · rw [← top_le_iff]
    /-
      case h.h
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : Algebra R S
      inst✝¹ : Algebra R T
      inst✝ : Algebra.EssFiniteType R S
      f g : AlgHom R S T
      H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
      ⊢ LE.le Top.top (Algebra.adjoin R (setOf fun x => Membership.mem (Algebra.EssF …
    -/
    rintro ⟨x, hx⟩ _
    /-
      case h.h.mk
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : Algebra R S
      inst✝¹ : Algebra R T
      inst✝ : Algebra.EssFiniteType R S
      f g : AlgHom R S T
      H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
      x : S
      hx : Membership.mem (Algebra.EssFiniteType.subalgebra R S) x
      a✝ : Membership.mem Top.top ⟨x, hx⟩
      ⊢ Membership.mem (Algebra.adjoin R (setOf fun x => Membership.mem (Algebra.Ess …
    -/
    refine Algebra.adjoin_induction ?_ ?_ ?_ ?_ hx
      /-
        case h.h.mk.refine_1
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : Algebra R S
        inst✝¹ : Algebra R T
        inst✝ : Algebra.EssFiniteType R S
        f g : AlgHom R S T
        H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
        x : S
        hx : Membership.mem (Algebra.EssFiniteType.subalgebra R S) x
        a✝ : Membership.mem Top.top ⟨x, hx⟩
        ⊢ ∀ (x : S) (hx : Membership.mem (↑(Algebra.EssFiniteType.finset R S)) x), Mem …
      -/
    · intro x hx; exact Algebra.subset_adjoin hx
                  /-
                    🎉 no goals
                  -/
      /-
        case h.h.mk.refine_2
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : Algebra R S
        inst✝¹ : Algebra R T
        inst✝ : Algebra.EssFiniteType R S
        f g : AlgHom R S T
        H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
        x : S
        hx : Membership.mem (Algebra.EssFiniteType.subalgebra R S) x
        a✝ : Membership.mem Top.top ⟨x, hx⟩
        ⊢ ∀ (r : R), Membership.mem (Algebra.adjoin R (setOf fun x => Membership.mem ( …
      -/
    · intro r; exact Subalgebra.algebraMap_mem _ _
               /-
                 🎉 no goals
               -/
      /-
        case h.h.mk.refine_3
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : Algebra R S
        inst✝¹ : Algebra R T
        inst✝ : Algebra.EssFiniteType R S
        f g : AlgHom R S T
        H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
        x : S
        hx : Membership.mem (Algebra.EssFiniteType.subalgebra R S) x
        a✝ : Membership.mem Top.top ⟨x, hx⟩
        ⊢ ∀ (x y : S) (hx : Membership.mem (Algebra.adjoin R ↑(Algebra.EssFiniteType.f …
      -/
    · intro x y _ _ hx hy; exact add_mem hx hy
                           /-
                             🎉 no goals
                           -/
      /-
        case h.h.mk.refine_4
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁵ : CommRing R
        inst✝⁴ : CommRing S
        inst✝³ : CommRing T
        inst✝² : Algebra R S
        inst✝¹ : Algebra R T
        inst✝ : Algebra.EssFiniteType R S
        f g : AlgHom R S T
        H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
        x : S
        hx : Membership.mem (Algebra.EssFiniteType.subalgebra R S) x
        a✝ : Membership.mem Top.top ⟨x, hx⟩
        ⊢ ∀ (x y : S) (hx : Membership.mem (Algebra.adjoin R ↑(Algebra.EssFiniteType.f …
      -/
    · intro x y _ _ hx hy; exact mul_mem hx hy
                           /-
                             🎉 no goals
                           -/
    /-
      case h.hs
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : CommRing T
      inst✝² : Algebra R S
      inst✝¹ : Algebra R T
      inst✝ : Algebra.EssFiniteType R S
      f g : AlgHom R S T
      H : ∀ (s : S), Membership.mem (Algebra.EssFiniteType.finset R S) s → Eq (f s)  …
      ⊢ Set.EqOn (⇑(f.comp (IsScalarTower.toAlgHom R (Subtype fun x => Membership.me …
    -/
  · rintro ⟨x, hx⟩ hx'; exact H x hx'
                        /-
                          🎉 no goals
                        -/


