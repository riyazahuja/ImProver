/-- The submonoid of `S = M⁻¹R` consisting of `{ 1 / x | x ∈ M }`. -/
def invSubmonoid : Submonoid S :=
  (M.map (algebraMap R S)).leftInv


theorem submonoid_map_le_is_unit : M.map (algebraMap R S) ≤ IsUnit.submonoid S := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ LE.le (Submonoid.map (algebraMap R S) M) (IsUnit.submonoid S)
  -/
  rintro _ ⟨a, ha, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : R
    ha : Membership.mem (↑M) a
    ⊢ Membership.mem (IsUnit.submonoid S) ((algebraMap R S) a)
  -/
  exact IsLocalization.map_units S ⟨_, ha⟩
  /-
    🎉 no goals
  -/


/-- There is an equivalence of monoids between the image of `M` and `invSubmonoid`. -/
noncomputable abbrev equivInvSubmonoid : M.map (algebraMap R S) ≃* invSubmonoid M S :=
  ((M.map (algebraMap R S)).leftInvEquiv (submonoid_map_le_is_unit M S)).symm


/-- There is a canonical map from `M` to `invSubmonoid` sending `x` to `1 / x`. -/
noncomputable def toInvSubmonoid : M →* invSubmonoid M S :=
  (equivInvSubmonoid M S).toMonoidHom.comp ((algebraMap R S : R →* S).submonoidMap M)


theorem toInvSubmonoid_surjective : Function.Surjective (toInvSubmonoid M S) :=
  Function.Surjective.comp (β := M.map (algebraMap R S))
    (Equiv.surjective (equivInvSubmonoid _ _).toEquiv) (MonoidHom.submonoidMap_surjective _ _)


@[simp]
theorem toInvSubmonoid_mul (m : M) : (toInvSubmonoid M S m : S) * algebraMap R S m = 1 :=
  Submonoid.leftInvEquiv_symm_mul _ (submonoid_map_le_is_unit _ _) _


@[simp]
theorem mul_toInvSubmonoid (m : M) : algebraMap R S m * (toInvSubmonoid M S m : S) = 1 :=
  Submonoid.mul_leftInvEquiv_symm _ (submonoid_map_le_is_unit _ _) ⟨_, _⟩


@[simp]
theorem smul_toInvSubmonoid (m : M) : m • (toInvSubmonoid M S m : S) = 1 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (HSMul.hSMul m ↑((IsLocalization.toInvSubmonoid M S) m)) 1
  -/
  convert mul_toInvSubmonoid M S m
  /-
    case h.e'_2.h.e
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (HSMul.hSMul m) (HMul.hMul ((algebraMap R S) ↑m))
  -/
  ext
  /-
    case h.e'_2.h.e.h
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    m : Subtype fun x => Membership.mem M x
    x✝ : S
    ⊢ Eq (HSMul.hSMul m x✝) (HMul.hMul ((algebraMap R S) ↑m) x✝)
  -/
  rw [← Algebra.smul_def]
  /-
    case h.e'_2.h.e.h
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    m : Subtype fun x => Membership.mem M x
    x✝ : S
    ⊢ Eq (HSMul.hSMul m x✝) (HSMul.hSMul (↑m) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem surj'' (z : S) : ∃ (r : R) (m : M), z = r • (toInvSubmonoid M S m : S) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    ⊢ Exists fun r => Exists fun m => Eq z (HSMul.hSMul r ↑((IsLocalization.toInvS …
  -/
  rcases IsLocalization.surj M z with ⟨⟨r, m⟩, e : z * _ = algebraMap R S r⟩
  /-
    case intro.mk
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    r : R
    m : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMap …
    ⊢ Exists fun r => Exists fun m => Eq z (HSMul.hSMul r ↑((IsLocalization.toInvS …
  -/
  refine ⟨r, m, ?_⟩
  /-
    case intro.mk
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    r : R
    m : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMap …
    ⊢ Eq z (HSMul.hSMul r ↑((IsLocalization.toInvSubmonoid M S) m))
  -/
  rw [Algebra.smul_def, ← e, mul_assoc]
  /-
    case intro.mk
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    r : R
    m : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul z ((algebraMap R S) ↑{ fst := r, snd := m }.2)) ((algebraMap …
    ⊢ Eq z (HMul.hMul z (HMul.hMul ((algebraMap R S) ↑{ fst := r, snd := m }.2) ↑( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toInvSubmonoid_eq_mk' (x : M) : (toInvSubmonoid M S x : S) = mk' S 1 x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : Subtype fun x => Membership.mem M x
    ⊢ Eq (↑((IsLocalization.toInvSubmonoid M S) x)) (IsLocalization.mk' S 1 x)
  -/
  rw [← (IsLocalization.map_units S x).mul_left_inj]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : Subtype fun x => Membership.mem M x
    ⊢ Eq (HMul.hMul (↑((IsLocalization.toInvSubmonoid M S) x)) ((algebraMap R S) ↑ …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_invSubmonoid_iff_exists_mk' (x : S) :
    x ∈ invSubmonoid M S ↔ ∃ m : M, mk' S 1 m = x := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : S
    ⊢ Iff (Membership.mem (IsLocalization.invSubmonoid M S) x) (Exists fun m => Eq …
  -/
  simp_rw [← toInvSubmonoid_eq_mk']
  exact ⟨fun h => ⟨_, congr_arg Subtype.val (toInvSubmonoid_surjective M S ⟨x, h⟩).choose_spec⟩,
    fun h => h.choose_spec ▸ (toInvSubmonoid M S h.choose).prop⟩


theorem span_invSubmonoid : Submodule.span R (invSubmonoid M S : Set S) = ⊤ := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ Eq (Submodule.span R ↑(IsLocalization.invSubmonoid M S)) Top.top
  -/
  rw [eq_top_iff]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ LE.le Top.top (Submodule.span R ↑(IsLocalization.invSubmonoid M S))
  -/
  rintro x -
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : S
    ⊢ Membership.mem (Submodule.span R ↑(IsLocalization.invSubmonoid M S)) x
  -/
  rcases IsLocalization.surj'' M x with ⟨r, m, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    r : R
    m : Subtype fun x => Membership.mem M x
    ⊢ Membership.mem (Submodule.span R ↑(IsLocalization.invSubmonoid M S)) (HSMul. …
  -/
  exact Submodule.smul_mem _ _ (Submodule.subset_span (toInvSubmonoid M S m).prop)
  /-
    🎉 no goals
  -/


theorem finiteType_of_monoid_fg [Monoid.FG M] : Algebra.FiniteType R S := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    ⊢ Algebra.FiniteType R S
  -/
  have := Monoid.fg_of_surjective _ (toInvSubmonoid_surjective M S)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    this : Monoid.FG (Subtype fun x => Membership.mem (IsLocalization.invSubmonoid …
    ⊢ Algebra.FiniteType R S
  -/
  rw [Monoid.fg_iff_submonoid_fg] at this
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    this : (IsLocalization.invSubmonoid M S).FG
    ⊢ Algebra.FiniteType R S
  -/
  rcases this with ⟨s, hs⟩
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    s : Finset S
    hs : Eq (Submonoid.closure ↑s) (IsLocalization.invSubmonoid M S)
    ⊢ Algebra.FiniteType R S
  -/
  refine ⟨⟨s, ?_⟩⟩
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    s : Finset S
    hs : Eq (Submonoid.closure ↑s) (IsLocalization.invSubmonoid M S)
    ⊢ Eq (Algebra.adjoin R ↑s) Top.top
  -/
  rw [eq_top_iff]
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    s : Finset S
    hs : Eq (Submonoid.closure ↑s) (IsLocalization.invSubmonoid M S)
    ⊢ LE.le Top.top (Algebra.adjoin R ↑s)
  -/
  rintro x -
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    s : Finset S
    hs : Eq (Submonoid.closure ↑s) (IsLocalization.invSubmonoid M S)
    x : S
    ⊢ Membership.mem (Algebra.adjoin R ↑s) x
  -/
  change x ∈ (Subalgebra.toSubmodule (Algebra.adjoin R _ : Subalgebra R S) : Set S)
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    s : Finset S
    hs : Eq (Submonoid.closure ↑s) (IsLocalization.invSubmonoid M S)
    x : S
    ⊢ Membership.mem (↑(Subalgebra.toSubmodule (Algebra.adjoin R ↑s))) x
  -/
  rw [Algebra.adjoin_eq_span, hs, span_invSubmonoid]
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    inst✝ : Monoid.FG (Subtype fun x => Membership.mem M x)
    s : Finset S
    hs : Eq (Submonoid.closure ↑s) (IsLocalization.invSubmonoid M S)
    x : S
    ⊢ Membership.mem (↑Top.top) x
  -/
  trivial
  /-
    🎉 no goals
  -/


