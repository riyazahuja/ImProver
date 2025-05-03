/-- The typeclass `IsLocalization (M : Submonoid R) S` where `S` is an `R`-algebra
expresses that `S` is isomorphic to the localization of `R` at `M`. -/
@[mk_iff] class IsLocalization : Prop where
  -- Porting note: add ' to fields, and made new versions of these with either `S` or `M` explicit.
  /-- Everything in the image of `algebraMap` is a unit -/
  map_units' : ∀ y : M, IsUnit (algebraMap R S y)
  /-- The `algebraMap` is surjective -/
  surj' : ∀ z : S, ∃ x : R × M, z * algebraMap R S x.2 = algebraMap R S x.1
  /-- The kernel of `algebraMap` is contained in the annihilator of `M`;
      it is then equal to the annihilator by `map_units'` -/
  exists_of_eq : ∀ {x y}, algebraMap R S x = algebraMap R S y → ∃ c : M, ↑c * x = ↑c * y


@[inherit_doc IsLocalization.map_units']
theorem map_units : ∀ y : M, IsUnit (algebraMap R S y) :=
  IsLocalization.map_units'


@[inherit_doc IsLocalization.surj']
theorem surj : ∀ z : S, ∃ x : R × M, z * algebraMap R S x.2 = algebraMap R S x.1 :=
  IsLocalization.surj'


@[inherit_doc IsLocalization.exists_of_eq]
theorem eq_iff_exists {x y} : algebraMap R S x = algebraMap R S y ↔ ∃ c : M, ↑c * x = ↑c * y :=
  Iff.intro IsLocalization.exists_of_eq fun ⟨c, h⟩ ↦ by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x y : R
      x✝ : Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      c : Subtype fun x => Membership.mem M x
      h : Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y)
    -/
    apply_fun algebraMap R S at h
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x y : R
      x✝ : Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      c : Subtype fun x => Membership.mem M x
      h : Eq ((algebraMap R S) (HMul.hMul (↑c) x)) ((algebraMap R S) (HMul.hMul (↑c) …
      ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y)
    -/
    rw [map_mul, map_mul] at h
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x y : R
      x✝ : Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      c : Subtype fun x => Membership.mem M x
      h : Eq (HMul.hMul ((algebraMap R S) ↑c) ((algebraMap R S) x)) (HMul.hMul ((alg …
      ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y)
    -/
    exact (IsLocalization.map_units S c).mul_right_inj.mp h
    /-
      🎉 no goals
    -/


theorem of_le (N : Submonoid R) (h₁ : M ≤ N) (h₂ : ∀ r ∈ N, IsUnit (algebraMap R S r)) :
    IsLocalization N S where
  map_units' r := h₂ r r.2
  surj' s :=
    have ⟨⟨x, y, hy⟩, H⟩ := IsLocalization.surj M s
    ⟨⟨x, y, h₁ hy⟩, H⟩
  exists_of_eq {x y} := by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      N : Submonoid R
      h₁ : LE.le M N
      h₂ : ∀ (r : R), Membership.mem N r → IsUnit ((algebraMap R S) r)
      x y : R
      ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y) → Exists fun c => Eq (HMul.hMul …
    -/
    rw [IsLocalization.eq_iff_exists M]
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      N : Submonoid R
      h₁ : LE.le M N
      h₂ : ∀ (r : R), Membership.mem N r → IsUnit ((algebraMap R S) r)
      x y : R
      ⊢ (Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)) → Exists fun c => …
    -/
    rintro ⟨c, hc⟩
    /-
      case intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      N : Submonoid R
      h₁ : LE.le M N
      h₂ : ∀ (r : R), Membership.mem N r → IsUnit ((algebraMap R S) r)
      x y : R
      c : Subtype fun x => Membership.mem M x
      hc : Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) y)
    -/
    exact ⟨⟨c, h₁ c.2⟩, hc⟩
    /-
      🎉 no goals
    -/


theorem of_le_of_exists_dvd (N : Submonoid R) (h₁ : M ≤ N) (h₂ : ∀ n ∈ N, ∃ m ∈ M, n ∣ m) :
    IsLocalization N S :=
  of_le M N h₁ fun n hn ↦ have ⟨m, hm, dvd⟩ := h₂ n hn
    isUnit_of_dvd_unit (map_dvd _ dvd) (map_units S ⟨m, hm⟩)


/-- `IsLocalization.toLocalizationWithZeroMap M S` shows `S` is the monoid localization of
`R` at `M`. -/
@[simps]
def toLocalizationWithZeroMap : Submonoid.LocalizationWithZeroMap M S where
  __ := algebraMap R S
  toFun := algebraMap R S
  map_units' := IsLocalization.map_units _
  surj' := IsLocalization.surj _
  exists_of_eq _ _ := IsLocalization.exists_of_eq


/-- `IsLocalization.toLocalizationMap M S` shows `S` is the monoid localization of `R` at `M`. -/
abbrev toLocalizationMap : Submonoid.LocalizationMap M S :=
  (toLocalizationWithZeroMap M S).toLocalizationMap


@[simp]
theorem toLocalizationMap_toMap : (toLocalizationMap M S).toMap = (algebraMap R S : R →*₀ S) :=
  rfl


theorem toLocalizationMap_toMap_apply (x) : (toLocalizationMap M S).toMap x = algebraMap R S x :=
  rfl


theorem surj₂ : ∀ z w : S, ∃ z' w' : R, ∃ d : M,
    (z * algebraMap R S d = algebraMap R S z') ∧ (w * algebraMap R S d = algebraMap R S w') :=
  (toLocalizationMap M S).surj₂


/-- Given a localization map `f : M →* N`, a section function sending `z : N` to some
`(x, y) : M × S` such that `f x * (f y)⁻¹ = z`. -/
noncomputable def sec (z : S) : R × M :=
  Classical.choose <| IsLocalization.surj _ z


@[simp]
theorem toLocalizationMap_sec : (toLocalizationMap M S).sec = sec M :=
  rfl


/-- Given `z : S`, `IsLocalization.sec M z` is defined to be a pair `(x, y) : R × M` such
that `z * f y = f x` (so this lemma is true by definition). -/
theorem sec_spec (z : S) :
    z * algebraMap R S (IsLocalization.sec M z).2 = algebraMap R S (IsLocalization.sec M z).1 :=
  Classical.choose_spec <| IsLocalization.surj _ z


/-- Given `z : S`, `IsLocalization.sec M z` is defined to be a pair `(x, y) : R × M` such
that `z * f y = f x`, so this lemma is just an application of `S`'s commutativity. -/
theorem sec_spec' (z : S) :
    algebraMap R S (IsLocalization.sec M z).1 = algebraMap R S (IsLocalization.sec M z).2 * z := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    ⊢ Eq ((algebraMap R S) (IsLocalization.sec M z).1) (HMul.hMul ((algebraMap R S …
  -/
  rw [mul_comm, sec_spec]
  /-
    🎉 no goals
  -/


/-- If `M` contains `0` then the localization at `M` is trivial. -/
theorem subsingleton (h : 0 ∈ M) : Subsingleton S := (toLocalizationMap M S).subsingleton h


theorem map_right_cancel {x y} {c : M} (h : algebraMap R S (c * x) = algebraMap R S (c * y)) :
    algebraMap R S x = algebraMap R S y :=
  (toLocalizationMap M S).map_right_cancel h


theorem map_left_cancel {x y} {c : M} (h : algebraMap R S (x * c) = algebraMap R S (y * c)) :
    algebraMap R S x = algebraMap R S y :=
  (toLocalizationMap M S).map_left_cancel h


theorem eq_zero_of_fst_eq_zero {z x} {y : M} (h : z * algebraMap R S y = algebraMap R S x)
    (hx : x = 0) : z = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    x : R
    y : Subtype fun x => Membership.mem M x
    h : Eq (HMul.hMul z ((algebraMap R S) ↑y)) ((algebraMap R S) x)
    hx : Eq x 0
    ⊢ Eq z 0
  -/
  rw [hx, (algebraMap R S).map_zero] at h
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    z : S
    x : R
    y : Subtype fun x => Membership.mem M x
    h : Eq (HMul.hMul z ((algebraMap R S) ↑y)) 0
    hx : Eq x 0
    ⊢ Eq z 0
  -/
  exact (IsUnit.mul_left_eq_zero (IsLocalization.map_units S y)).1 h
  /-
    🎉 no goals
  -/


theorem map_eq_zero_iff (r : R) : algebraMap R S r = 0 ↔ ∃ m : M, ↑m * r = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    r : R
    ⊢ Iff (Eq ((algebraMap R S) r) 0) (Exists fun m => Eq (HMul.hMul (↑m) r) 0)
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
      r : R
      ⊢ Eq ((algebraMap R S) r) 0 → Exists fun m => Eq (HMul.hMul (↑m) r) 0
    -/
  · intro h
    /-
      case mp
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      r : R
      h : Eq ((algebraMap R S) r) 0
      ⊢ Exists fun m => Eq (HMul.hMul (↑m) r) 0
    -/
    obtain ⟨m, hm⟩ := (IsLocalization.eq_iff_exists M S).mp ((algebraMap R S).map_zero.trans h.symm)
    /-
      case mp.intro
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      r : R
      h : Eq ((algebraMap R S) r) 0
      m : Subtype fun x => Membership.mem M x
      hm : Eq (HMul.hMul (↑m) 0) (HMul.hMul (↑m) r)
      ⊢ Exists fun m => Eq (HMul.hMul (↑m) r) 0
    -/
    exact ⟨m, by simpa using hm.symm⟩
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
      r : R
      ⊢ (Exists fun m => Eq (HMul.hMul (↑m) r) 0) → Eq ((algebraMap R S) r) 0
    -/
  · rintro ⟨m, hm⟩
    rw [← (IsLocalization.map_units S m).mul_right_inj, mul_zero, ← RingHom.map_mul, hm,
      RingHom.map_zero]


/-- `IsLocalization.mk' S` is the surjection sending `(x, y) : R × M` to
`f x * (f y)⁻¹`. -/
noncomputable def mk' (x : R) (y : M) : S :=
  (toLocalizationMap M S).mk' x y


@[simp]
theorem mk'_sec (z : S) : mk' S (IsLocalization.sec M z).1 (IsLocalization.sec M z).2 = z :=
  (toLocalizationMap M S).mk'_sec _


theorem mk'_mul (x₁ x₂ : R) (y₁ y₂ : M) : mk' S (x₁ * x₂) (y₁ * y₂) = mk' S x₁ y₁ * mk' S x₂ y₂ :=
  (toLocalizationMap M S).mk'_mul _ _ _ _


theorem mk'_one (x) : mk' S x (1 : M) = algebraMap R S x :=
  (toLocalizationMap M S).mk'_one _


@[simp]
theorem mk'_spec (x) (y : M) : mk' S x y * algebraMap R S y = algebraMap R S x :=
  (toLocalizationMap M S).mk'_spec _ _


@[simp]
theorem mk'_spec' (x) (y : M) : algebraMap R S y * mk' S x y = algebraMap R S x :=
  (toLocalizationMap M S).mk'_spec' _ _


@[simp]
theorem mk'_spec_mk (x) (y : R) (hy : y ∈ M) :
    mk' S x ⟨y, hy⟩ * algebraMap R S y = algebraMap R S x :=
  mk'_spec S x ⟨y, hy⟩


@[simp]
theorem mk'_spec'_mk (x) (y : R) (hy : y ∈ M) :
    algebraMap R S y * mk' S x ⟨y, hy⟩ = algebraMap R S x :=
  mk'_spec' S x ⟨y, hy⟩


theorem eq_mk'_iff_mul_eq {x} {y : M} {z} :
    z = mk' S x y ↔ z * algebraMap R S y = algebraMap R S x :=
  (toLocalizationMap M S).eq_mk'_iff_mul_eq


theorem eq_mk'_of_mul_eq {x : R} {y : M} {z : R} (h : z * y = x) : (algebraMap R S) z = mk' S x y :=
                            /-
                              R : Type u_1
                              inst✝³ : CommSemiring R
                              M : Submonoid R
                              S : Type u_2
                              inst✝² : CommSemiring S
                              inst✝¹ : Algebra R S
                              inst✝ : IsLocalization M S
                              x : R
                              y : Subtype fun x => Membership.mem M x
                              z : R
                              h : Eq (HMul.hMul z ↑y) x
                              ⊢ Eq (HMul.hMul ((algebraMap R S) z) ((algebraMap R S) ↑y)) ((algebraMap R S) x)
                            -/
  eq_mk'_iff_mul_eq.mpr (by rw [← h, map_mul])
                            /-
                              🎉 no goals
                            -/


theorem mk'_eq_iff_eq_mul {x} {y : M} {z} :
    mk' S x y = z ↔ algebraMap R S x = z * algebraMap R S y :=
  (toLocalizationMap M S).mk'_eq_iff_eq_mul


theorem mk'_add_eq_iff_add_mul_eq_mul {x} {y : M} {z₁ z₂} :
    mk' S x y + z₁ = z₂ ↔ algebraMap R S x + z₁ * algebraMap R S y = z₂ * algebraMap R S y := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    y : Subtype fun x => Membership.mem M x
    z₁ z₂ : S
    ⊢ Iff (Eq (HAdd.hAdd (IsLocalization.mk' S x y) z₁) z₂) (Eq (HAdd.hAdd ((algeb …
  -/
  rw [← mk'_spec S x y, ← IsUnit.mul_left_inj (IsLocalization.map_units S y), right_distrib]
  /-
    🎉 no goals
  -/


theorem mk'_pow (x : R) (y : M) (n : ℕ) : mk' S (x ^ n) (y ^ n) = mk' S x y ^ n := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    y : Subtype fun x => Membership.mem M x
    n : Nat
    ⊢ Eq (IsLocalization.mk' S (HPow.hPow x n) (HPow.hPow y n)) (HPow.hPow (IsLoca …
  -/
  simp_rw [IsLocalization.mk'_eq_iff_eq_mul, SubmonoidClass.coe_pow, map_pow, ← mul_pow]
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    y : Subtype fun x => Membership.mem M x
    n : Nat
    ⊢ Eq (HPow.hPow ((algebraMap R S) x) n) (HPow.hPow (HMul.hMul (IsLocalization. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mk'_surjective (z : S) : ∃ (x : _) (y : M), mk' S x y = z :=
  let ⟨r, hr⟩ := IsLocalization.surj _ z
  ⟨r.1, r.2, (eq_mk'_iff_mul_eq.2 hr).symm⟩


/-- The localization of a `Fintype` is a `Fintype`. Cannot be an instance. -/
noncomputable def fintype' [Fintype R] : Fintype S :=
  have := Classical.propDecidable
  Fintype.ofSurjective (Function.uncurry <| IsLocalization.mk' S) fun a =>
    Prod.exists'.mpr <| IsLocalization.mk'_surjective M a


/-- Localizing at a submonoid with 0 inside it leads to the trivial ring. -/
def uniqueOfZeroMem (h : (0 : R) ∈ M) : Unique S :=
                          /-
                            R : Type u_1
                            inst✝⁴ : CommSemiring R
                            M : Submonoid R
                            S : Type u_2
                            inst✝³ : CommSemiring S
                            inst✝² : Algebra R S
                            P : Type u_3
                            inst✝¹ : CommSemiring P
                            inst✝ : IsLocalization M S
                            h : Membership.mem M 0
                            ⊢ Eq 0 1
                          -/
  uniqueOfZeroEqOne <| by simpa using IsLocalization.map_units S ⟨0, h⟩
                          /-
                            🎉 no goals
                          -/


theorem mk'_eq_iff_eq {x₁ x₂} {y₁ y₂ : M} :
    mk' S x₁ y₁ = mk' S x₂ y₂ ↔ algebraMap R S (y₂ * x₁) = algebraMap R S (y₁ * x₂) :=
  (toLocalizationMap M S).mk'_eq_iff_eq


theorem mk'_eq_iff_eq' {x₁ x₂} {y₁ y₂ : M} :
    mk' S x₁ y₁ = mk' S x₂ y₂ ↔ algebraMap R S (x₁ * y₂) = algebraMap R S (x₂ * y₁) :=
  (toLocalizationMap M S).mk'_eq_iff_eq'


protected theorem eq {a₁ b₁} {a₂ b₂ : M} :
    mk' S a₁ a₂ = mk' S b₁ b₂ ↔ ∃ c : M, ↑c * (↑b₂ * a₁) = c * (a₂ * b₁) :=
  (toLocalizationMap M S).eq


theorem mk'_eq_zero_iff (x : R) (s : M) : mk' S x s = 0 ↔ ∃ m : M, ↑m * x = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    s : Subtype fun x => Membership.mem M x
    ⊢ Iff (Eq (IsLocalization.mk' S x s) 0) (Exists fun m => Eq (HMul.hMul (↑m) x) …
  -/
  rw [← (map_units S s).mul_left_inj, mk'_spec, zero_mul, map_eq_zero_iff M]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk'_zero (s : M) : IsLocalization.mk' S 0 s = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    s : Subtype fun x => Membership.mem M x
    ⊢ Eq (IsLocalization.mk' S 0 s) 0
  -/
  rw [eq_comm, IsLocalization.eq_mk'_iff_mul_eq, zero_mul, map_zero]
  /-
    🎉 no goals
  -/


theorem ne_zero_of_mk'_ne_zero {x : R} {y : M} (hxy : IsLocalization.mk' S x y ≠ 0) : x ≠ 0 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    y : Subtype fun x => Membership.mem M x
    hxy : Ne (IsLocalization.mk' S x y) 0
    ⊢ Ne x 0
  -/
  rintro rfl
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    y : Subtype fun x => Membership.mem M x
    hxy : Ne (IsLocalization.mk' S 0 y) 0
    ⊢ False
  -/
  exact hxy (IsLocalization.mk'_zero _)
  /-
    🎉 no goals
  -/


theorem eq_iff_eq [Algebra R P] [IsLocalization M P] {x y} :
    algebraMap R S x = algebraMap R S y ↔ algebraMap R P x = algebraMap R P y :=
  (toLocalizationMap M S).eq_iff_eq (toLocalizationMap M P)


theorem mk'_eq_iff_mk'_eq [Algebra R P] [IsLocalization M P] {x₁ x₂} {y₁ y₂ : M} :
    mk' S x₁ y₁ = mk' S x₂ y₂ ↔ mk' P x₁ y₁ = mk' P x₂ y₂ :=
  (toLocalizationMap M S).mk'_eq_iff_mk'_eq (toLocalizationMap M P)


theorem mk'_eq_of_eq {a₁ b₁ : R} {a₂ b₂ : M} (H : ↑a₂ * b₁ = ↑b₂ * a₁) :
    mk' S a₁ a₂ = mk' S b₁ b₂ :=
  (toLocalizationMap M S).mk'_eq_of_eq H


theorem mk'_eq_of_eq' {a₁ b₁ : R} {a₂ b₂ : M} (H : b₁ * ↑a₂ = a₁ * ↑b₂) :
    mk' S a₁ a₂ = mk' S b₁ b₂ :=
  (toLocalizationMap M S).mk'_eq_of_eq' H


theorem mk'_cancel (a : R) (b c : M) :
    mk' S (a * c) (b * c) = mk' S a b := (toLocalizationMap M S).mk'_cancel _ _ _


@[simp]
theorem mk'_self {x : R} (hx : x ∈ M) : mk' S x ⟨x, hx⟩ = 1 :=
  (toLocalizationMap M S).mk'_self _ hx


@[simp]
theorem mk'_self' {x : M} : mk' S (x : R) x = 1 :=
  (toLocalizationMap M S).mk'_self' _


theorem mk'_self'' {x : M} : mk' S x.1 x = 1 :=
  mk'_self' _


theorem mul_mk'_eq_mk'_of_mul (x y : R) (z : M) :
    (algebraMap R S) x * mk' S y z = mk' S (x * y) z :=
  (toLocalizationMap M S).mul_mk'_eq_mk'_of_mul _ _ _


theorem mk'_eq_mul_mk'_one (x : R) (y : M) : mk' S x y = (algebraMap R S) x * mk' S 1 y :=
  ((toLocalizationMap M S).mul_mk'_one_eq_mk' _ _).symm


@[simp]
theorem mk'_mul_cancel_left (x : R) (y : M) : mk' S (y * x : R) y = (algebraMap R S) x :=
  (toLocalizationMap M S).mk'_mul_cancel_left _ _


theorem mk'_mul_cancel_right (x : R) (y : M) : mk' S (x * y) y = (algebraMap R S) x :=
  (toLocalizationMap M S).mk'_mul_cancel_right _ _


@[simp]
theorem mk'_mul_mk'_eq_one (x y : M) : mk' S (x : R) y * mk' S (y : R) x = 1 := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x y : Subtype fun x => Membership.mem M x
    ⊢ Eq (HMul.hMul (IsLocalization.mk' S (↑x) y) (IsLocalization.mk' S (↑y) x)) 1
  -/
  rw [← mk'_mul, mul_comm]; exact mk'_self _ _
                            /-
                              🎉 no goals
                            -/


theorem mk'_mul_mk'_eq_one' (x : R) (y : M) (h : x ∈ M) : mk' S x y * mk' S (y : R) ⟨x, h⟩ = 1 :=
  mk'_mul_mk'_eq_one ⟨x, h⟩ _


theorem smul_mk' (x y : R) (m : M) : x • mk' S y m = mk' S (x * y) m := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x y : R
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (HSMul.hSMul x (IsLocalization.mk' S y m)) (IsLocalization.mk' S (HMul.hM …
  -/
  nth_rw 2 [← one_mul m]
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x y : R
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (HSMul.hSMul x (IsLocalization.mk' S y m)) (IsLocalization.mk' S (HMul.hM …
  -/
  rw [mk'_mul, mk'_one, Algebra.smul_def]
  /-
    🎉 no goals
  -/


@[simp] theorem smul_mk'_one (x : R) (m : M) : x • mk' S 1 m = mk' S x m := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (HSMul.hSMul x (IsLocalization.mk' S 1 m)) (IsLocalization.mk' S x m)
  -/
  rw [smul_mk', mul_one]
  /-
    🎉 no goals
  -/


@[simp] lemma smul_mk'_self {m : M} {r : R} :
    (m : R) • mk' S r m = algebraMap R S r := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    m : Subtype fun x => Membership.mem M x
    r : R
    ⊢ Eq (HSMul.hSMul (↑m) (IsLocalization.mk' S r m)) ((algebraMap R S) r)
  -/
  rw [smul_mk', mk'_mul_cancel_left]
  /-
    🎉 no goals
  -/


@[simps]
instance invertible_mk'_one (s : M) : Invertible (IsLocalization.mk' S (1 : R) s) where
  invOf := algebraMap R S s
                       /-
                         R : Type u_1
                         inst✝⁴ : CommSemiring R
                         M : Submonoid R
                         S : Type u_2
                         inst✝³ : CommSemiring S
                         inst✝² : Algebra R S
                         P : Type u_3
                         inst✝¹ : CommSemiring P
                         inst✝ : IsLocalization M S
                         s : Subtype fun x => Membership.mem M x
                         ⊢ Eq (HMul.hMul ((algebraMap R S) ↑s) (IsLocalization.mk' S 1 s)) 1
                       -/
  invOf_mul_self := by simp
                       /-
                         🎉 no goals
                       -/
                       /-
                         R : Type u_1
                         inst✝⁴ : CommSemiring R
                         M : Submonoid R
                         S : Type u_2
                         inst✝³ : CommSemiring S
                         inst✝² : Algebra R S
                         P : Type u_3
                         inst✝¹ : CommSemiring P
                         inst✝ : IsLocalization M S
                         s : Subtype fun x => Membership.mem M x
                         ⊢ Eq (HMul.hMul (IsLocalization.mk' S 1 s) ((algebraMap R S) ↑s)) 1
                       -/
  mul_invOf_self := by simp
                       /-
                         🎉 no goals
                       -/


theorem isUnit_comp (j : S →+* P) (y : M) : IsUnit (j.comp (algebraMap R S) y) :=
  (toLocalizationMap M S).isUnit_comp j.toMonoidHom _


/-- Given a localization map `f : R →+* S` for a submonoid `M ⊆ R` and a map of `CommSemiring`s
`g : R →+* P` such that `g(M) ⊆ Units P`, `f x = f y → g x = g y` for all `x y : R`. -/
theorem eq_of_eq {g : R →+* P} (hg : ∀ y : M, IsUnit (g y)) {x y}
    (h : (algebraMap R S) x = (algebraMap R S) y) : g x = g y :=
  Submonoid.LocalizationMap.eq_of_eq (toLocalizationMap M S) (g := g.toMonoidHom) hg h


theorem mk'_add (x₁ x₂ : R) (y₁ y₂ : M) :
    mk' S (x₁ * y₂ + x₂ * y₁) (y₁ * y₂) = mk' S x₁ y₁ + mk' S x₂ y₂ :=
  mk'_eq_iff_eq_mul.2 <|
    Eq.symm
      (by
        rw [mul_comm (_ + _), mul_add, mul_mk'_eq_mk'_of_mul, mk'_add_eq_iff_add_mul_eq_mul,
          mul_comm (_ * _), ← mul_assoc, add_comm, ← map_mul, mul_mk'_eq_mk'_of_mul,
          mk'_add_eq_iff_add_mul_eq_mul]
        /-
          R : Type u_1
          inst✝³ : CommSemiring R
          M : Submonoid R
          S : Type u_2
          inst✝² : CommSemiring S
          inst✝¹ : Algebra R S
          inst✝ : IsLocalization M S
          x₁ x₂ : R
          y₁ y₂ : Subtype fun x => Membership.mem M x
          ⊢ Eq (HAdd.hAdd ((algebraMap R S) (HMul.hMul (HMul.hMul ↑y₁ ↑(HMul.hMul y₁ y₂) …
        -/
        simp only [map_add, Submonoid.coe_mul, map_mul]
        /-
          R : Type u_1
          inst✝³ : CommSemiring R
          M : Submonoid R
          S : Type u_2
          inst✝² : CommSemiring S
          inst✝¹ : Algebra R S
          inst✝ : IsLocalization M S
          x₁ x₂ : R
          y₁ y₂ : Subtype fun x => Membership.mem M x
          ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul ((algebraMap R S) ↑y₁) (HMul.hMul ((alge …
        -/
        ring)
        /-
          🎉 no goals
        -/


theorem mul_add_inv_left {g : R →+* P} (h : ∀ y : M, IsUnit (g y)) (y : M) (w z₁ z₂ : P) :
    w * ↑(IsUnit.liftRight (g.toMonoidHom.restrict M) h y)⁻¹ + z₁ =
    z₂ ↔ w + g y * z₁ = g y * z₂ := by
  rw [mul_comm, ← one_mul z₁, ← Units.inv_mul (IsUnit.liftRight (g.toMonoidHom.restrict M) h y),
    mul_assoc, ← mul_add, Units.inv_mul_eq_iff_eq_mul, Units.inv_mul_cancel_left,
    IsUnit.coe_liftRight]
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    M : Submonoid R
    P : Type u_3
    inst✝ : CommSemiring P
    g : RingHom R P
    h : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
    y : Subtype fun x => Membership.mem M x
    w z₁ z₂ : P
    ⊢ Iff (Eq (HAdd.hAdd w (HMul.hMul (((↑g).restrict M) y) z₁)) (HMul.hMul (((↑g) …
  -/
  simp [RingHom.toMonoidHom_eq_coe, MonoidHom.restrict_apply]
  /-
    🎉 no goals
  -/


theorem lift_spec_mul_add {g : R →+* P} (hg : ∀ y : M, IsUnit (g y)) (z w w' v) :
    ((toLocalizationWithZeroMap M S).lift g.toMonoidWithZeroHom hg) z * w + w' = v ↔
      g ((toLocalizationMap M S).sec z).1 * w + g ((toLocalizationMap M S).sec z).2 * w' =
        g ((toLocalizationMap M S).sec z).2 * v := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    P : Type u_3
    inst✝¹ : CommSemiring P
    inst✝ : IsLocalization M S
    g : RingHom R P
    hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
    z : S
    w w' v : P
    ⊢ Iff (Eq (HAdd.hAdd (HMul.hMul (((IsLocalization.toLocalizationWithZeroMap M  …
  -/
  erw [mul_comm, ← mul_assoc, mul_add_inv_left hg, mul_comm]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    P : Type u_3
    inst✝¹ : CommSemiring P
    inst✝ : IsLocalization M S
    g : RingHom R P
    hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
    z : S
    w w' v : P
    ⊢ Iff (Eq (HAdd.hAdd (HMul.hMul (↑g.toMonoidWithZeroHom ((IsLocalization.toLoc …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a localization map `f : R →+* S` for a submonoid `M ⊆ R` and a map of `CommSemiring`s
`g : R →+* P` such that `g y` is invertible for all `y : M`, the homomorphism induced from
`S` to `P` sending `z : S` to `g x * (g y)⁻¹`, where `(x, y) : R × M` are such that
`z = f x * (f y)⁻¹`. -/
noncomputable def lift {g : R →+* P} (hg : ∀ y : M, IsUnit (g y)) : S →+* P :=
  { Submonoid.LocalizationWithZeroMap.lift (toLocalizationWithZeroMap M S)
      g.toMonoidWithZeroHom hg with
    map_add' := by
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        ⊢ ∀ (x y : S), Eq ((↑{ toFun := (↑__src✝).toFun, map_one' := ⋯, map_mul' := ⋯  …
      -/
      intro x y
      erw [(toLocalizationMap M S).lift_spec, mul_add, mul_comm, eq_comm, lift_spec_mul_add,
        add_comm, mul_comm, mul_assoc, mul_comm, mul_assoc, lift_spec_mul_add]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        x y : S
        ⊢ Eq (HAdd.hAdd (HMul.hMul (g ((IsLocalization.toLocalizationMap M S).sec y).1 …
      -/
      simp_rw [← mul_assoc]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        x y : S
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (g ((IsLocalization.toLocalizationMap M  …
      -/
      show g _ * g _ * g _ + g _ * g _ * g _ = g _ * g _ * g _
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        x y : S
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (g ((IsLocalization.toLocalizationMap M  …
      -/
      simp_rw [← map_mul g, ← map_add g]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        x y : S
        ⊢ Eq (g (HAdd.hAdd (HMul.hMul (HMul.hMul ((IsLocalization.toLocalizationMap M  …
      -/
      apply eq_of_eq (S := S) hg
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        x y : S
        ⊢ Eq ((algebraMap R S) (HAdd.hAdd (HMul.hMul (HMul.hMul ((IsLocalization.toLoc …
      -/
      simp only [sec_spec', toLocalizationMap_sec, map_add, map_mul]
      /-
        R : Type u_1
        inst✝⁴ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝³ : CommSemiring S
        inst✝² : Algebra R S
        P : Type u_3
        inst✝¹ : CommSemiring P
        inst✝ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        x y : S
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul ((algebraMap R S) ↑(IsLocaliz …
      -/
      ring }
      /-
        🎉 no goals
      -/


/-- Given a localization map `f : R →+* S` for a submonoid `M ⊆ R` and a map of `CommSemiring`s
`g : R →* P` such that `g y` is invertible for all `y : M`, the homomorphism induced from
`S` to `P` maps `f x * (f y)⁻¹` to `g x * (g y)⁻¹` for all `x : R, y ∈ M`. -/
theorem lift_mk' (x y) :
    lift hg (mk' S x y) = g x * ↑(IsUnit.liftRight (g.toMonoidHom.restrict M) hg y)⁻¹ :=
  (toLocalizationMap M S).lift_mk' _ _ _


theorem lift_mk'_spec (x v) (y : M) : lift hg (mk' S x y) = v ↔ g x = g y * v :=
  (toLocalizationMap M S).lift_mk'_spec _ _ _ _


@[simp]
theorem lift_eq (x : R) : lift hg ((algebraMap R S) x) = g x :=
  (toLocalizationMap M S).lift_eq _ _


theorem lift_eq_iff {x y : R × M} :
    lift hg (mk' S x.1 x.2) = lift hg (mk' S y.1 y.2) ↔ g (x.1 * y.2) = g (y.1 * x.2) :=
  (toLocalizationMap M S).lift_eq_iff _


@[simp]
theorem lift_comp : (lift hg).comp (algebraMap R S) = g :=
  RingHom.ext <| (DFunLike.ext_iff (F := MonoidHom _ _)).1 <| (toLocalizationMap M S).lift_comp _


@[simp]
theorem lift_of_comp (j : S →+* P) : lift (isUnit_comp M j) = j :=
  RingHom.ext <| (DFunLike.ext_iff (F := MonoidHom _ _)).1 <|
    (toLocalizationMap M S).lift_of_comp j.toMonoidHom


/-- See note [partially-applied ext lemmas] -/
theorem monoidHom_ext ⦃j k : S →* P⦄
    (h : j.comp (algebraMap R S : R →* S) = k.comp (algebraMap R S)) : j = k :=
  Submonoid.LocalizationMap.epic_of_localizationMap (toLocalizationMap M S) <| DFunLike.congr_fun h


/-- See note [partially-applied ext lemmas] -/
theorem ringHom_ext ⦃j k : S →+* P⦄ (h : j.comp (algebraMap R S) = k.comp (algebraMap R S)) :
    j = k :=
  RingHom.coe_monoidHom_injective <| monoidHom_ext M <| MonoidHom.ext <| RingHom.congr_fun h


/-- To show `j` and `k` agree on the whole localization, it suffices to show they agree
on the image of the base ring, if they preserve `1` and `*`. -/
protected theorem ext (j k : S → P) (hj1 : j 1 = 1) (hk1 : k 1 = 1)
    (hjm : ∀ a b, j (a * b) = j a * j b) (hkm : ∀ a b, k (a * b) = k a * k b)
    (h : ∀ a, j (algebraMap R S a) = k (algebraMap R S a)) : j = k :=
  let j' : MonoidHom S P :=
    { toFun := j, map_one' := hj1, map_mul' := hjm }
  let k' : MonoidHom S P :=
    { toFun := k, map_one' := hk1, map_mul' := hkm }
  have : j' = k' := monoidHom_ext M (MonoidHom.ext h)
                              /-
                                R : Type u_1
                                inst✝⁴ : CommSemiring R
                                M : Submonoid R
                                S : Type u_2
                                inst✝³ : CommSemiring S
                                inst✝² : Algebra R S
                                P : Type u_3
                                inst✝¹ : CommSemiring P
                                inst✝ : IsLocalization M S
                                j k : S → P
                                hj1 : Eq (j 1) 1
                                hk1 : Eq (k 1) 1
                                hjm : ∀ (a b : S), Eq (j (HMul.hMul a b)) (HMul.hMul (j a) (j b))
                                hkm : ∀ (a b : S), Eq (k (HMul.hMul a b)) (HMul.hMul (k a) (k b))
                                h : ∀ (a : R), Eq (j ((algebraMap R S) a)) (k ((algebraMap R S) a))
                                j' : MonoidHom S P := { toFun := j, map_one' := hj1, map_mul' := hjm }
                                k' : MonoidHom S P := { toFun := k, map_one' := hk1, map_mul' := hkm }
                                this : Eq j' k'
                                ⊢ Eq (↑j').toFun (↑k').toFun
                              -/
  show j'.toFun = k'.toFun by rw [this]
                              /-
                                🎉 no goals
                              -/

theorem lift_unique {j : S →+* P} (hj : ∀ x, j ((algebraMap R S) x) = g x) : lift hg = j :=
  RingHom.ext <|
    (DFunLike.ext_iff (F := MonoidHom _ _)).1 <|
      Submonoid.LocalizationMap.lift_unique (toLocalizationMap M S) (g := g.toMonoidHom) hg
        (j := j.toMonoidHom) hj


@[simp]
theorem lift_id (x) : lift (map_units S : ∀ _ : M, IsUnit _) x = x :=
  (toLocalizationMap M S).lift_id _


theorem lift_surjective_iff :
    Surjective (lift hg : S → P) ↔ ∀ v : P, ∃ x : R × M, v * g x.2 = g x.1 :=
  (toLocalizationMap M S).lift_surjective_iff hg


theorem lift_injective_iff :
    Injective (lift hg : S → P) ↔ ∀ x y, algebraMap R S x = algebraMap R S y ↔ g x = g y :=
  (toLocalizationMap M S).lift_injective_iff hg


variable (M) in
include M in
lemma injective_iff_map_algebraMap_eq {T} [CommRing T] (f : S →+* T) :
    Function.Injective f ↔ ∀ x y,
      algebraMap R S x = algebraMap R S y ↔ f (algebraMap R S x) = f (algebraMap R S y) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_4
    inst✝ : CommRing T
    f : RingHom S T
    ⊢ Iff (Function.Injective ⇑f) (∀ (x y : R), Iff (Eq ((algebraMap R S) x) ((alg …
  -/
  rw [← IsLocalization.lift_of_comp (M := M) f, IsLocalization.lift_injective_iff]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_4
    inst✝ : CommRing T
    f : RingHom S T
    ⊢ Iff (∀ (x y : R), Iff (Eq ((algebraMap R S) x) ((algebraMap R S) y)) (Eq ((f …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Map a homomorphism `g : R →+* P` to `S →+* Q`, where `S` and `Q` are
localizations of `R` and `P` at `M` and `T` respectively,
such that `g(M) ⊆ T`.

We send `z : S` to `algebraMap P Q (g x) * (algebraMap P Q (g y))⁻¹`, where
`(x, y) : R × M` are such that `z = f x * (f y)⁻¹`. -/
noncomputable def map (g : R →+* P) (hy : M ≤ T.comap g) : S →+* Q :=
  lift (M := M) (g := (algebraMap P Q).comp g) fun y => map_units _ ⟨g y, hy y.2⟩


@[simp]
theorem map_eq (x) : map Q g hy ((algebraMap R S) x) = algebraMap P Q (g x) :=
  lift_eq (fun y => map_units _ ⟨g y, hy y.2⟩) x


@[simp]
theorem map_comp : (map Q g hy).comp (algebraMap R S) = (algebraMap P Q).comp g :=
  lift_comp fun y => map_units _ ⟨g y, hy y.2⟩


theorem map_mk' (x) (y : M) : map Q g hy (mk' S x y) = mk' Q (g x) ⟨g y, hy y.2⟩ :=
  Submonoid.LocalizationMap.map_mk' (toLocalizationMap M S) (g := g.toMonoidHom)
    (fun y => hy y.2) (k := toLocalizationMap T Q) ..


theorem map_unique (j : S →+* Q) (hj : ∀ x : R, j (algebraMap R S x) = algebraMap P Q (g x)) :
    map Q g hy = j :=
  lift_unique (fun y => map_units _ ⟨g y, hy y.2⟩) hj


/-- If `CommSemiring` homs `g : R →+* P, l : P →+* A` induce maps of localizations, the composition
of the induced maps equals the map of localizations induced by `l ∘ g`. -/
theorem map_comp_map {A : Type*} [CommSemiring A] {U : Submonoid A} {W} [CommSemiring W]
    [Algebra A W] [IsLocalization U W] {l : P →+* A} (hl : T ≤ U.comap l) :
    (map W l hl).comp (map Q g hy : S →+* _) = map W (l.comp g) fun _ hx => hl (hy hx) :=
  RingHom.ext fun x =>
    Submonoid.LocalizationMap.map_map (P := P) (toLocalizationMap M S) (fun y => hy y.2)
      (toLocalizationMap U W) (fun w => hl w.2) x


/-- If `CommSemiring` homs `g : R →+* P, l : P →+* A` induce maps of localizations, the composition
of the induced maps equals the map of localizations induced by `l ∘ g`. -/
theorem map_map {A : Type*} [CommSemiring A] {U : Submonoid A} {W} [CommSemiring W] [Algebra A W]
    [IsLocalization U W] {l : P →+* A} (hl : T ≤ U.comap l) (x : S) :
    map W l hl (map Q g hy x) = map W (l.comp g) (fun _ hx => hl (hy hx)) x := by
  /-
    R : Type u_1
    inst✝¹¹ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝¹⁰ : CommSemiring S
    inst✝⁹ : Algebra R S
    P : Type u_3
    inst✝⁸ : CommSemiring P
    inst✝⁷ : IsLocalization M S
    g : RingHom R P
    T : Submonoid P
    Q : Type u_4
    inst✝⁶ : CommSemiring Q
    inst✝⁵ : Algebra P Q
    inst✝⁴ : IsLocalization T Q
    hy : LE.le M (Submonoid.comap g T)
    A : Type u_5
    inst✝³ : CommSemiring A
    U : Submonoid A
    W : Type u_6
    inst✝² : CommSemiring W
    inst✝¹ : Algebra A W
    inst✝ : IsLocalization U W
    l : RingHom P A
    hl : LE.le T (Submonoid.comap l U)
    x : S
    ⊢ Eq ((IsLocalization.map W l hl) ((IsLocalization.map Q g hy) x)) ((IsLocaliz …
  -/
  rw [← map_comp_map (Q := Q) hy hl]; rfl
                                      /-
                                        🎉 no goals
                                      -/


protected theorem map_smul (x : S) (z : R) : map Q g hy (z • x : S) = g z • map Q g hy x := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommSemiring S
    inst✝⁵ : Algebra R S
    P : Type u_3
    inst✝⁴ : CommSemiring P
    inst✝³ : IsLocalization M S
    g : RingHom R P
    T : Submonoid P
    Q : Type u_4
    inst✝² : CommSemiring Q
    inst✝¹ : Algebra P Q
    inst✝ : IsLocalization T Q
    hy : LE.le M (Submonoid.comap g T)
    x : S
    z : R
    ⊢ Eq ((IsLocalization.map Q g hy) (HSMul.hSMul z x)) (HSMul.hSMul (g z) ((IsLo …
  -/
  rw [Algebra.smul_def, Algebra.smul_def, RingHom.map_mul, map_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_id_mk' {Q : Type*} [CommSemiring Q] [Algebra R Q] [IsLocalization M Q] (x) (y : M) :
    map Q (RingHom.id R) (le_refl M) (mk' S x y) = mk' Q x y :=
  map_mk' ..


@[simp]
theorem map_id (z : S) (h : M ≤ M.comap (RingHom.id R) := le_refl M) :
    map S (RingHom.id _) h z = z :=
  lift_id _


/-- If `S`, `Q` are localizations of `R` and `P` at submonoids `M, T` respectively, an
isomorphism `j : R ≃+* P` such that `j(M) = T` induces an isomorphism of localizations
`S ≃+* Q`. -/
@[simps]
noncomputable def ringEquivOfRingEquiv (h : R ≃+* P) (H : M.map h.toMonoidHom = T) : S ≃+* Q :=
  have H' : T.map h.symm.toMonoidHom = M := by
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      P : Type u_3
      inst✝⁴ : CommSemiring P
      inst✝³ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      T : Submonoid P
      Q : Type u_4
      inst✝² : CommSemiring Q
      inst✝¹ : Algebra P Q
      inst✝ : IsLocalization T Q
      h : RingEquiv R P
      H : Eq (Submonoid.map h.toMonoidHom M) T
      ⊢ Eq (Submonoid.map h.symm.toMonoidHom T) M
    -/
    rw [← M.map_id, ← H, Submonoid.map_map]
    /-
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      P : Type u_3
      inst✝⁴ : CommSemiring P
      inst✝³ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      T : Submonoid P
      Q : Type u_4
      inst✝² : CommSemiring Q
      inst✝¹ : Algebra P Q
      inst✝ : IsLocalization T Q
      h : RingEquiv R P
      H : Eq (Submonoid.map h.toMonoidHom M) T
      ⊢ Eq (Submonoid.map (h.symm.toMonoidHom.comp h.toMonoidHom) M) (Submonoid.map  …
    -/
    congr
    /-
      case e_f
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      P : Type u_3
      inst✝⁴ : CommSemiring P
      inst✝³ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      T : Submonoid P
      Q : Type u_4
      inst✝² : CommSemiring Q
      inst✝¹ : Algebra P Q
      inst✝ : IsLocalization T Q
      h : RingEquiv R P
      H : Eq (Submonoid.map h.toMonoidHom M) T
      ⊢ Eq (h.symm.toMonoidHom.comp h.toMonoidHom) (MonoidHom.id R)
    -/
    ext
    /-
      case e_f.h
      R : Type u_1
      inst✝⁷ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝⁶ : CommSemiring S
      inst✝⁵ : Algebra R S
      P : Type u_3
      inst✝⁴ : CommSemiring P
      inst✝³ : IsLocalization M S
      g : RingHom R P
      hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
      T : Submonoid P
      Q : Type u_4
      inst✝² : CommSemiring Q
      inst✝¹ : Algebra P Q
      inst✝ : IsLocalization T Q
      h : RingEquiv R P
      H : Eq (Submonoid.map h.toMonoidHom M) T
      x✝ : R
      ⊢ Eq ((h.symm.toMonoidHom.comp h.toMonoidHom) x✝) ((MonoidHom.id R) x✝)
    -/
    apply h.symm_apply_apply
    /-
      🎉 no goals
    -/
  { map Q (h : R →+* P) (M.le_comap_of_map_le (le_of_eq H)) with
    toFun := map Q (h : R →+* P) (M.le_comap_of_map_le (le_of_eq H))
    invFun := map S (h.symm : P →+* R) (T.le_comap_of_map_le (le_of_eq H'))
    left_inv := fun x => by
      /-
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        P : Type u_3
        inst✝⁴ : CommSemiring P
        inst✝³ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        T : Submonoid P
        Q : Type u_4
        inst✝² : CommSemiring Q
        inst✝¹ : Algebra P Q
        inst✝ : IsLocalization T Q
        h : RingEquiv R P
        H : Eq (Submonoid.map h.toMonoidHom M) T
        H' : Eq (Submonoid.map h.symm.toMonoidHom T) M
        x : S
        ⊢ Eq ((IsLocalization.map S ↑h.symm ⋯) ((IsLocalization.map Q ↑h ⋯) x)) x
      -/
      rw [map_map, map_unique _ (RingHom.id _), RingHom.id_apply]
      /-
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        P : Type u_3
        inst✝⁴ : CommSemiring P
        inst✝³ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        T : Submonoid P
        Q : Type u_4
        inst✝² : CommSemiring Q
        inst✝¹ : Algebra P Q
        inst✝ : IsLocalization T Q
        h : RingEquiv R P
        H : Eq (Submonoid.map h.toMonoidHom M) T
        H' : Eq (Submonoid.map h.symm.toMonoidHom T) M
        x : S
        ⊢ ∀ (x : R), Eq ((RingHom.id S) ((algebraMap R S) x)) ((algebraMap R S) (((↑h. …
      -/
      simp
      /-
        🎉 no goals
      -/
    right_inv := fun x => by
      /-
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        P : Type u_3
        inst✝⁴ : CommSemiring P
        inst✝³ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        T : Submonoid P
        Q : Type u_4
        inst✝² : CommSemiring Q
        inst✝¹ : Algebra P Q
        inst✝ : IsLocalization T Q
        h : RingEquiv R P
        H : Eq (Submonoid.map h.toMonoidHom M) T
        H' : Eq (Submonoid.map h.symm.toMonoidHom T) M
        x : Q
        ⊢ Eq ((IsLocalization.map Q ↑h ⋯) ((IsLocalization.map S ↑h.symm ⋯) x)) x
      -/
      rw [map_map, map_unique _ (RingHom.id _), RingHom.id_apply]
      /-
        R : Type u_1
        inst✝⁷ : CommSemiring R
        M : Submonoid R
        S : Type u_2
        inst✝⁶ : CommSemiring S
        inst✝⁵ : Algebra R S
        P : Type u_3
        inst✝⁴ : CommSemiring P
        inst✝³ : IsLocalization M S
        g : RingHom R P
        hg : ∀ (y : Subtype fun x => Membership.mem M x), IsUnit (g ↑y)
        T : Submonoid P
        Q : Type u_4
        inst✝² : CommSemiring Q
        inst✝¹ : Algebra P Q
        inst✝ : IsLocalization T Q
        h : RingEquiv R P
        H : Eq (Submonoid.map h.toMonoidHom M) T
        H' : Eq (Submonoid.map h.symm.toMonoidHom T) M
        x : Q
        ⊢ ∀ (x : P), Eq ((RingHom.id Q) ((algebraMap P Q) x)) ((algebraMap P Q) (((↑h) …
      -/
      simp }
      /-
        🎉 no goals
      -/


theorem ringEquivOfRingEquiv_eq_map {j : R ≃+* P} (H : M.map j.toMonoidHom = T) :
    (ringEquivOfRingEquiv S Q j H : S →+* Q) =
      map Q (j : R →+* P) (M.le_comap_of_map_le (le_of_eq H)) :=
  rfl


theorem ringEquivOfRingEquiv_eq {j : R ≃+* P} (H : M.map j.toMonoidHom = T) (x) :
    ringEquivOfRingEquiv S Q j H ((algebraMap R S) x) = algebraMap P Q (j x) := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommSemiring S
    inst✝⁵ : Algebra R S
    P : Type u_3
    inst✝⁴ : CommSemiring P
    inst✝³ : IsLocalization M S
    T : Submonoid P
    Q : Type u_4
    inst✝² : CommSemiring Q
    inst✝¹ : Algebra P Q
    inst✝ : IsLocalization T Q
    j : RingEquiv R P
    H : Eq (Submonoid.map j.toMonoidHom M) T
    x : R
    ⊢ Eq ((IsLocalization.ringEquivOfRingEquiv S Q j H) ((algebraMap R S) x)) ((al …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ringEquivOfRingEquiv_mk' {j : R ≃+* P} (H : M.map j.toMonoidHom = T) (x : R) (y : M) :
    ringEquivOfRingEquiv S Q j H (mk' S x y) =
      mk' Q (j x) ⟨j y, show j y ∈ T from H ▸ Set.mem_image_of_mem j y.2⟩ := by
  /-
    R : Type u_1
    inst✝⁷ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝⁶ : CommSemiring S
    inst✝⁵ : Algebra R S
    P : Type u_3
    inst✝⁴ : CommSemiring P
    inst✝³ : IsLocalization M S
    T : Submonoid P
    Q : Type u_4
    inst✝² : CommSemiring Q
    inst✝¹ : Algebra P Q
    inst✝ : IsLocalization T Q
    j : RingEquiv R P
    H : Eq (Submonoid.map j.toMonoidHom M) T
    x : R
    y : Subtype fun x => Membership.mem M x
    ⊢ Eq ((IsLocalization.ringEquivOfRingEquiv S Q j H) (IsLocalization.mk' S x y) …
  -/
  simp [map_mk']
  /-
    🎉 no goals
  -/


@[simp]
theorem ringEquivOfRingEquiv_symm {j : R ≃+* P} (H : M.map j.toMonoidHom = T) :
    (ringEquivOfRingEquiv S Q j H).symm =
      ringEquivOfRingEquiv Q S j.symm (show T.map j.symm.toMonoidHom = M by
        erw [← H, ← Submonoid.comap_equiv_eq_map_symm,
          Submonoid.comap_map_eq_of_injective j.injective]) := rfl


lemma at_units (S : Submonoid R)
    (hS : S ≤ IsUnit.submonoid R) : IsLocalization S R where
  map_units' y := hS y.prop
                               /-
                                 R : Type u_1
                                 inst✝ : CommSemiring R
                                 S : Submonoid R
                                 hS : LE.le S (IsUnit.submonoid R)
                                 s : R
                                 ⊢ Eq (HMul.hMul s ((algebraMap R R) ↑{ fst := s, snd := 1 }.2)) ((algebraMap R …
                               -/
  surj' := fun s ↦ ⟨⟨s, 1⟩, by simp⟩
                               /-
                                 🎉 no goals
                               -/
  exists_of_eq := fun {x y} (e : x = y) ↦ ⟨1, e ▸ rfl⟩


/-- Injectivity of a map descends to the map induced on localizations. -/
theorem map_injective_of_injective (h : Function.Injective g) [IsLocalization (M.map g) Q] :
    Function.Injective (map Q g M.le_comap_map : S → Q) :=
  (toLocalizationMap M S).map_injective_of_injective h (toLocalizationMap (M.map g) Q)


/-- Surjectivity of a map descends to the map induced on localizations. -/
theorem map_surjective_of_surjective (h : Function.Surjective g) [IsLocalization (M.map g) Q] :
    Function.Surjective (map Q g M.le_comap_map : S → Q) :=
  (toLocalizationMap M S).map_surjective_of_surjective h (toLocalizationMap (M.map g) Q)


theorem isLocalization_of_base_ringEquiv [IsLocalization M S] (h : R ≃+* P) :
    haveI := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    IsLocalization (M.map h.toMonoidHom) S := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    P : Type u_3
    inst✝¹ : CommSemiring P
    inst✝ : IsLocalization M S
    h : RingEquiv R P
    ⊢ IsLocalization (Submonoid.map h.toMonoidHom M) S
  -/
  letI : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    P : Type u_3
    inst✝¹ : CommSemiring P
    inst✝ : IsLocalization M S
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    ⊢ IsLocalization (Submonoid.map h.toMonoidHom M) S
  -/
  constructor
    /-
      case map_units'
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      ⊢ ∀ (y : Subtype fun x => Membership.mem (Submonoid.map h.toMonoidHom M) x), I …
    -/
  · rintro ⟨_, ⟨y, hy, rfl⟩⟩
    /-
      case map_units'.mk.intro.intro
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : R
      hy : Membership.mem (↑M) y
      ⊢ IsUnit ((algebraMap P S) ↑⟨h.toMonoidHom y, ⋯⟩)
    -/
    convert IsLocalization.map_units S ⟨y, hy⟩
    /-
      case h.e'_3
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : R
      hy : Membership.mem (↑M) y
      ⊢ Eq ((algebraMap P S) ↑⟨h.toMonoidHom y, ⋯⟩) ((algebraMap R S) ↑⟨y, hy⟩)
    -/
    dsimp only [RingHom.algebraMap_toAlgebra, RingHom.comp_apply]
    /-
      case h.e'_3
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : R
      hy : Membership.mem (↑M) y
      ⊢ Eq ((algebraMap R S) (h.symm.toRingHom (h.toMonoidHom y))) ((algebraMap R S) …
    -/
    exact congr_arg _ (h.symm_apply_apply _)
    /-
      🎉 no goals
    -/
    /-
      case surj'
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      ⊢ ∀ (z : S), Exists fun x => Eq (HMul.hMul z ((algebraMap P S) ↑x.2)) ((algebr …
    -/
  · intro y
    /-
      case surj'
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : S
      ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap P S) ↑x.2)) ((algebraMap P S) x …
    -/
    obtain ⟨⟨x, s⟩, e⟩ := IsLocalization.surj M y
    /-
      case surj'.intro.mk
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMap …
      ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap P S) ↑x.2)) ((algebraMap P S) x …
    -/
    refine ⟨⟨h x, _, _, s.prop, rfl⟩, ?_⟩
    /-
      case surj'.intro.mk
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul y ((algebraMap R S) ↑{ fst := x, snd := s }.2)) ((algebraMap …
      ⊢ Eq (HMul.hMul y ((algebraMap P S) ↑{ fst := h x, snd := ⟨h.toMonoidHom ↑s, ⋯ …
    -/
    dsimp only [RingHom.algebraMap_toAlgebra, RingHom.comp_apply] at e ⊢
    /-
      case surj'.intro.mk
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      y : S
      x : R
      s : Subtype fun x => Membership.mem M x
      e : Eq (HMul.hMul y ((algebraMap R S) ↑s)) ((algebraMap R S) x)
      ⊢ Eq (HMul.hMul y ((algebraMap R S) (h.symm.toRingHom (h.toMonoidHom ↑s)))) (( …
    -/
                  /-
                    🎉 no goals
                  -/
    convert e <;> exact h.symm_apply_apply _
                  /-
                    🎉 no goals
                  -/
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      ⊢ ∀ {x y : P}, Eq ((algebraMap P S) x) ((algebraMap P S) y) → Exists fun c =>  …
    -/
  · intro x y
    rw [RingHom.algebraMap_toAlgebra, RingHom.comp_apply, RingHom.comp_apply,
      IsLocalization.eq_iff_exists M S]
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      x y : P
      ⊢ (Exists fun c => Eq (HMul.hMul (↑c) (h.symm.toRingHom x)) (HMul.hMul (↑c) (h …
    -/
    simp_rw [← h.toEquiv.apply_eq_iff_eq]
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      x y : P
      ⊢ (Exists fun c => Eq (h.toEquiv (HMul.hMul (↑c) (h.symm.toRingHom x))) (h.toE …
    -/
    change (∃ c : M, h (c * h.symm x) = h (c * h.symm y)) → _
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      x y : P
      ⊢ (Exists fun c => Eq (h (HMul.hMul (↑c) (h.symm x))) (h (HMul.hMul (↑c) (h.sy …
    -/
    simp only [RingEquiv.apply_symm_apply, RingEquiv.map_mul]
    /-
      case exists_of_eq
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : IsLocalization M S
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      x y : P
      ⊢ (Exists fun c => Eq (HMul.hMul (h ↑c) x) (HMul.hMul (h ↑c) y)) → Exists fun  …
    -/
    exact fun ⟨c, e⟩ ↦ ⟨⟨_, _, c.prop, rfl⟩, e⟩
    /-
      🎉 no goals
    -/


theorem isLocalization_iff_of_base_ringEquiv (h : R ≃+* P) :
    IsLocalization M S ↔
      haveI := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      IsLocalization (M.map h.toMonoidHom) S := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    ⊢ Iff (IsLocalization M S) (IsLocalization (Submonoid.map h.toMonoidHom M) S)
  -/
  letI : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    ⊢ Iff (IsLocalization M S) (IsLocalization (Submonoid.map h.toMonoidHom M) S)
  -/
  refine ⟨fun _ => isLocalization_of_base_ringEquiv M S h, ?_⟩
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    ⊢ IsLocalization (Submonoid.map h.toMonoidHom M) S → IsLocalization M S
  -/
  intro H
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    H : IsLocalization (Submonoid.map h.toMonoidHom M) S
    ⊢ IsLocalization M S
  -/
  convert isLocalization_of_base_ringEquiv (Submonoid.map (RingEquiv.toMonoidHom h) M) S h.symm
    /-
      case h.e'_3
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommSemiring P
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      H : IsLocalization (Submonoid.map h.toMonoidHom M) S
      ⊢ Eq M (Submonoid.map h.symm.toMonoidHom (Submonoid.map h.toMonoidHom M))
    -/
  · erw [Submonoid.map_equiv_eq_comap_symm, Submonoid.comap_map_eq_of_injective]
    /-
      case h.e'_3.hf
      R : Type u_1
      inst✝³ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      P : Type u_3
      inst✝ : CommSemiring P
      h : RingEquiv R P
      this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
      H : IsLocalization (Submonoid.map h.toMonoidHom M) S
      ⊢ Function.Injective ⇑h.symm.toMulEquiv.symm.toMonoidHom
    -/
    exact h.toEquiv.injective
    /-
      🎉 no goals
    -/
  /-
    case h.e'_6
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    H : IsLocalization (Submonoid.map h.toMonoidHom M) S
    ⊢ Eq inst✝¹ ((algebraMap P S).comp h.symm.symm.toRingHom).toAlgebra
  -/
  rw [RingHom.algebraMap_toAlgebra, RingHom.comp_assoc]
  /-
    case h.e'_6
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    H : IsLocalization (Submonoid.map h.toMonoidHom M) S
    ⊢ Eq inst✝¹ ((algebraMap R S).comp (h.symm.toRingHom.comp h.symm.symm.toRingHo …
  -/
  simp only [RingHom.comp_id, RingEquiv.symm_symm, RingEquiv.symm_toRingHom_comp_toRingHom]
  /-
    case h.e'_6
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    H : IsLocalization (Submonoid.map h.toMonoidHom M) S
    ⊢ Eq inst✝¹ (algebraMap R S).toAlgebra
  -/
  apply Algebra.algebra_ext
  /-
    case h.e'_6.h
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    H : IsLocalization (Submonoid.map h.toMonoidHom M) S
    ⊢ ∀ (r : R), Eq ((algebraMap R S) r) ((algebraMap R S) r)
  -/
  intro r
  /-
    case h.e'_6.h
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    P : Type u_3
    inst✝ : CommSemiring P
    h : RingEquiv R P
    this : Algebra P S := ((algebraMap R S).comp h.symm.toRingHom).toAlgebra
    H : IsLocalization (Submonoid.map h.toMonoidHom M) S
    r : R
    ⊢ Eq ((algebraMap R S) r) ((algebraMap R S) r)
  -/
  rw [RingHom.algebraMap_toAlgebra]
  /-
    🎉 no goals
  -/


theorem nonZeroDivisors_le_comap [IsLocalization M S] :
    nonZeroDivisors R ≤ (nonZeroDivisors S).comap (algebraMap R S) := by
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ LE.le (nonZeroDivisors R) (Submonoid.comap (algebraMap R S) (nonZeroDivisors …
  -/
  rintro a ha b (e : b * algebraMap R S a = 0)
  /-
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    b : S
    e : Eq (HMul.hMul b ((algebraMap R S) a)) 0
    ⊢ Eq b 0
  -/
  obtain ⟨x, s, rfl⟩ := mk'_surjective M b
  rw [← @mk'_one R _ M, ← mk'_mul, ← (algebraMap R S).map_zero, ← @mk'_one R _ M,
    IsLocalization.eq] at e
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    x : R
    s : Subtype fun x => Membership.mem M x
    e : Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul (↑1) (HMul.hMul x a))) (HMul …
    ⊢ Eq (IsLocalization.mk' S x s) 0
  -/
  obtain ⟨c, e⟩ := e
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    x : R
    s c : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul (↑c) (HMul.hMul (↑1) (HMul.hMul x a))) (HMul.hMul (↑c) (HMul …
    ⊢ Eq (IsLocalization.mk' S x s) 0
  -/
  rw [mul_zero, mul_zero, Submonoid.coe_one, one_mul, ← mul_assoc] at e
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    x : R
    s c : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul (HMul.hMul (↑c) x) a) 0
    ⊢ Eq (IsLocalization.mk' S x s) 0
  -/
  rw [mk'_eq_zero_iff]
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝³ : CommSemiring R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    x : R
    s c : Subtype fun x => Membership.mem M x
    e : Eq (HMul.hMul (HMul.hMul (↑c) x) a) 0
    ⊢ Exists fun m => Eq (HMul.hMul (↑m) x) 0
  -/
  exact ⟨c, ha _ e⟩
  /-
    🎉 no goals
  -/


theorem map_nonZeroDivisors_le [IsLocalization M S] :
    (nonZeroDivisors R).map (algebraMap R S) ≤ nonZeroDivisors S :=
  Submonoid.map_le_iff_le_comap.mpr (nonZeroDivisors_le_comap M S)


instance instUniqueLocalization [Subsingleton R] : Unique (Localization M) where
  uniq a := by
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Submonoid R
      S : Type u_2
      inst✝³ : CommSemiring S
      inst✝² : Algebra R S
      P : Type u_3
      inst✝¹ : CommSemiring P
      inst✝ : Subsingleton R
      a : Localization M
      ⊢ Eq a Inhabited.default
    -/
    with_unfolding_all show a = mk 1 1
    exact Localization.induction_on a fun _ => by
      congr <;> apply Subsingleton.elim


theorem add_mk (a b c d) : (mk a b : Localization M) + mk c d =
    mk ((b : R) * c + (d : R) * a) (b * d) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    a : R
    b : Subtype fun x => Membership.mem M x
    c : R
    d : Subtype fun x => Membership.mem M x
    ⊢ Eq (HAdd.hAdd (Localization.mk a b) (Localization.mk c d)) (Localization.mk  …
  -/
  rw [add_comm (b * c) (d * a), mul_comm b d]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    a : R
    b : Subtype fun x => Membership.mem M x
    c : R
    d : Subtype fun x => Membership.mem M x
    ⊢ Eq (HAdd.hAdd (Localization.mk a b) (Localization.mk c d)) (Localization.mk  …
  -/
  exact OreLocalization.oreDiv_add_oreDiv
  /-
    🎉 no goals
  -/


theorem add_mk_self (a b c) : (mk a b : Localization M) + mk c b = mk (a + c) b := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    a : R
    b : Subtype fun x => Membership.mem M x
    c : R
    ⊢ Eq (HAdd.hAdd (Localization.mk a b) (Localization.mk c b)) (Localization.mk  …
  -/
  rw [add_mk, mk_eq_mk_iff, r_eq_r']
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    a : R
    b : Subtype fun x => Membership.mem M x
    c : R
    ⊢ (Localization.r' M) { fst := HAdd.hAdd (HMul.hMul (↑b) c) (HMul.hMul (↑b) a) …
  -/
  refine (r' M).symm ⟨1, ?_⟩
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    a : R
    b : Subtype fun x => Membership.mem M x
    c : R
    ⊢ Eq (HMul.hMul (↑1) (HMul.hMul ↑{ fst := HAdd.hAdd (HMul.hMul (↑b) c) (HMul.h …
  -/
  simp only [Submonoid.coe_one, Submonoid.coe_mul]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    a : R
    b : Subtype fun x => Membership.mem M x
    c : R
    ⊢ Eq (HMul.hMul 1 (HMul.hMul (HMul.hMul ↑b ↑b) (HAdd.hAdd a c))) (HMul.hMul 1  …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- For any given denominator `b : M`, the map `a ↦ a / b` is an `AddMonoidHom` from `R` to
  `Localization M`-/
@[simps]
def mkAddMonoidHom (b : M) : R →+ Localization M where
  toFun a := mk a b
  map_zero' := mk_zero _
  map_add' _ _ := (add_mk_self _ _ _).symm


theorem mk_sum {ι : Type*} (f : ι → R) (s : Finset ι) (b : M) :
    mk (∑ i ∈ s, f i) b = ∑ i ∈ s, mk (f i) b :=
  map_sum (mkAddMonoidHom b) f s


theorem mk_list_sum (l : List R) (b : M) : mk l.sum b = (l.map fun a => mk a b).sum :=
  map_list_sum (mkAddMonoidHom b) l


theorem mk_multiset_sum (l : Multiset R) (b : M) : mk l.sum b = (l.map fun a => mk a b).sum :=
  (mkAddMonoidHom b).map_multiset_sum l


instance isLocalization : IsLocalization M (Localization M) where
  map_units' := (Localization.monoidOf M).map_units
  surj' := (Localization.monoidOf M).surj
  exists_of_eq := (Localization.monoidOf M).eq_iff_exists.mp


@[simp]
theorem toLocalizationMap_eq_monoidOf : toLocalizationMap M (Localization M) = monoidOf M :=
  rfl


theorem monoidOf_eq_algebraMap (x) : (monoidOf M).toMap x = algebraMap R (Localization M) x :=
  rfl


theorem mk_one_eq_algebraMap (x) : mk x 1 = algebraMap R (Localization M) x :=
  rfl


theorem mk_eq_mk'_apply (x y) : mk x y = IsLocalization.mk' (Localization M) x y := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    M : Submonoid R
    x : R
    y : Subtype fun x => Membership.mem M x
    ⊢ Eq (Localization.mk x y) (IsLocalization.mk' (Localization M) x y)
  -/
  rw [mk_eq_monoidOf_mk'_apply, mk', toLocalizationMap_eq_monoidOf]
  /-
    🎉 no goals
  -/

-- Porting note: removed `simp`. Left hand side can be simplified; not clear what normal form should
--be.

theorem mk_eq_mk' : (mk : R → M → Localization M) = IsLocalization.mk' (Localization M) :=
  mk_eq_monoidOf_mk'


theorem mk_algebraMap {A : Type*} [CommSemiring A] [Algebra A R] (m : A) :
    mk (algebraMap A R m) 1 = algebraMap A (Localization M) m := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    M : Submonoid R
    A : Type u_4
    inst✝¹ : CommSemiring A
    inst✝ : Algebra A R
    m : A
    ⊢ Eq (Localization.mk ((algebraMap A R) m) 1) ((algebraMap A (Localization M)) …
  -/
  rw [mk_eq_mk', mk'_eq_iff_eq_mul, Submonoid.coe_one, map_one, mul_one]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem neg_mk (a b) : -(mk a b : Localization M) = mk (-a) b := OreLocalization.neg_def _ _


theorem sub_mk (a c) (b d) : (mk a b : Localization M) - mk c d =
    mk ((d : R) * a - b * c) (b * d) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    M : Submonoid R
    a c : R
    b d : Subtype fun x => Membership.mem M x
    ⊢ Eq (HSub.hSub (Localization.mk a b) (Localization.mk c d)) (Localization.mk  …
  -/
  rw [sub_eq_add_neg, neg_mk, add_mk, add_comm, mul_neg, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


include M in
lemma injective_of_map_algebraMap_zero {T} [CommRing T] (f : S →+* T)
    (h : ∀ x, f (algebraMap R S x) = 0 → algebraMap R S x = 0) :
    Function.Injective f := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_5
    inst✝ : CommRing T
    f : RingHom S T
    h : ∀ (x : R), Eq (f ((algebraMap R S) x)) 0 → Eq ((algebraMap R S) x) 0
    ⊢ Function.Injective ⇑f
  -/
  rw [IsLocalization.injective_iff_map_algebraMap_eq M]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_5
    inst✝ : CommRing T
    f : RingHom S T
    h : ∀ (x : R), Eq (f ((algebraMap R S) x)) 0 → Eq ((algebraMap R S) x) 0
    ⊢ ∀ (x y : R), Iff (Eq ((algebraMap R S) x) ((algebraMap R S) y)) (Eq (f ((alg …
  -/
  refine fun x y ↦ ⟨fun hz ↦ hz ▸ rfl, fun hz ↦ ?_⟩
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_5
    inst✝ : CommRing T
    f : RingHom S T
    h : ∀ (x : R), Eq (f ((algebraMap R S) x)) 0 → Eq ((algebraMap R S) x) 0
    x y : R
    hz : Eq (f ((algebraMap R S) x)) (f ((algebraMap R S) y))
    ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y)
  -/
  rw [← sub_eq_zero, ← map_sub, ← map_sub] at hz
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_5
    inst✝ : CommRing T
    f : RingHom S T
    h : ∀ (x : R), Eq (f ((algebraMap R S) x)) 0 → Eq ((algebraMap R S) x) 0
    x y : R
    hz : Eq (f ((algebraMap R S) (HSub.hSub x y))) 0
    ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y)
  -/
  apply h at hz
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : IsLocalization M S
    T : Type u_5
    inst✝ : CommRing T
    f : RingHom S T
    h : ∀ (x : R), Eq (f ((algebraMap R S) x)) 0 → Eq ((algebraMap R S) x) 0
    x y : R
    hz : Eq ((algebraMap R S) (HSub.hSub x y)) 0
    ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) y)
  -/
  rwa [map_sub, sub_eq_zero] at hz
  /-
    🎉 no goals
  -/


theorem to_map_eq_zero_iff {x : R} (hM : M ≤ nonZeroDivisors R) : algebraMap R S x = 0 ↔ x = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    hM : LE.le M (nonZeroDivisors R)
    ⊢ Iff (Eq ((algebraMap R S) x) 0) (Eq x 0)
  -/
  rw [← (algebraMap R S).map_zero]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    hM : LE.le M (nonZeroDivisors R)
    ⊢ Iff (Eq ((algebraMap R S) x) ((algebraMap R S) 0)) (Eq x 0)
  -/
  constructor <;> intro h
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      hM : LE.le M (nonZeroDivisors R)
      h : Eq ((algebraMap R S) x) ((algebraMap R S) 0)
      ⊢ Eq x 0
    -/
  · cases' (eq_iff_exists M S).mp h with c hc
    /-
      case mp.intro
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      hM : LE.le M (nonZeroDivisors R)
      h : Eq ((algebraMap R S) x) ((algebraMap R S) 0)
      c : Subtype fun x => Membership.mem M x
      hc : Eq (HMul.hMul (↑c) x) (HMul.hMul (↑c) 0)
      ⊢ Eq x 0
    -/
    rw [mul_zero, mul_comm] at hc
    /-
      case mp.intro
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      hM : LE.le M (nonZeroDivisors R)
      h : Eq ((algebraMap R S) x) ((algebraMap R S) 0)
      c : Subtype fun x => Membership.mem M x
      hc : Eq (HMul.hMul x ↑c) 0
      ⊢ Eq x 0
    -/
    exact hM c.2 x hc
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      x : R
      hM : LE.le M (nonZeroDivisors R)
      h : Eq x 0
      ⊢ Eq ((algebraMap R S) x) ((algebraMap R S) 0)
    -/
  · rw [h]
    /-
      🎉 no goals
    -/


protected theorem injective (hM : M ≤ nonZeroDivisors R) : Injective (algebraMap R S) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    hM : LE.le M (nonZeroDivisors R)
    ⊢ Function.Injective ⇑(algebraMap R S)
  -/
  rw [injective_iff_map_eq_zero (algebraMap R S)]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    hM : LE.le M (nonZeroDivisors R)
    ⊢ ∀ (a : R), Eq ((algebraMap R S) a) 0 → Eq a 0
  -/
  intro a ha
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    hM : LE.le M (nonZeroDivisors R)
    a : R
    ha : Eq ((algebraMap R S) a) 0
    ⊢ Eq a 0
  -/
  rwa [to_map_eq_zero_iff S hM] at ha
  /-
    🎉 no goals
  -/


protected theorem to_map_ne_zero_of_mem_nonZeroDivisors [Nontrivial R] (hM : M ≤ nonZeroDivisors R)
    {x : R} (hx : x ∈ nonZeroDivisors R) : algebraMap R S x ≠ 0 :=
  show (algebraMap R S).toMonoidWithZeroHom x ≠ 0 from
    map_ne_zero_of_mem_nonZeroDivisors (algebraMap R S) (IsLocalization.injective S hM) hx


theorem sec_snd_ne_zero [Nontrivial R] (hM : M ≤ nonZeroDivisors R) (x : S) :
    ((sec M x).snd : R) ≠ 0 :=
  nonZeroDivisors.coe_ne_zero ⟨(sec M x).snd.val, hM (sec M x).snd.property⟩


theorem sec_fst_ne_zero [Nontrivial R] [NoZeroDivisors S] (hM : M ≤ nonZeroDivisors R) {x : S}
    (hx : x ≠ 0) : (sec M x).fst ≠ 0 := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsLocalization M S
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors S
    hM : LE.le M (nonZeroDivisors R)
    x : S
    hx : Ne x 0
    ⊢ Ne (IsLocalization.sec M x).1 0
  -/
  have hsec := sec_spec M x
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsLocalization M S
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors S
    hM : LE.le M (nonZeroDivisors R)
    x : S
    hx : Ne x 0
    hsec : Eq (HMul.hMul x ((algebraMap R S) ↑(IsLocalization.sec M x).2)) ((algeb …
    ⊢ Ne (IsLocalization.sec M x).1 0
  -/
  intro hfst
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝⁴ : CommRing S
    inst✝³ : Algebra R S
    inst✝² : IsLocalization M S
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors S
    hM : LE.le M (nonZeroDivisors R)
    x : S
    hx : Ne x 0
    hsec : Eq (HMul.hMul x ((algebraMap R S) ↑(IsLocalization.sec M x).2)) ((algeb …
    hfst : Eq (IsLocalization.sec M x).1 0
    ⊢ False
  -/
  rw [hfst, map_zero, mul_eq_zero, _root_.map_eq_zero_iff] at hsec
    /-
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      inst✝² : IsLocalization M S
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors S
      hM : LE.le M (nonZeroDivisors R)
      x : S
      hx : Ne x 0
      hsec : Or (Eq x 0) (Eq (↑(IsLocalization.sec M x).2) 0)
      hfst : Eq (IsLocalization.sec M x).1 0
      ⊢ False
    -/
  · exact Or.elim hsec hx (sec_snd_ne_zero hM x)
    /-
      🎉 no goals
    -/
    /-
      case hf
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Submonoid R
      S : Type u_2
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      inst✝² : IsLocalization M S
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors S
      hM : LE.le M (nonZeroDivisors R)
      x : S
      hx : Ne x 0
      hsec : Or (Eq x 0) (Eq ((algebraMap R S) ↑(IsLocalization.sec M x).2) 0)
      hfst : Eq (IsLocalization.sec M x).1 0
      ⊢ Function.Injective ⇑(algebraMap R S)
    -/
  · exact IsLocalization.injective S hM
    /-
      🎉 no goals
    -/


/-- A `CommRing` `S` which is the localization of a ring `R` without zero divisors at a subset of
non-zero elements does not have zero divisors. -/
theorem noZeroDivisors_of_le_nonZeroDivisors [Algebra A S] {M : Submonoid A} [IsLocalization M S]
    (hM : M ≤ nonZeroDivisors A) : NoZeroDivisors S :=
  { eq_zero_or_eq_zero_of_mul_eq_zero := by
      /-
        S : Type u_2
        inst✝⁴ : CommRing S
        A : Type u_6
        inst✝³ : CommRing A
        inst✝² : IsDomain A
        inst✝¹ : Algebra A S
        M : Submonoid A
        inst✝ : IsLocalization M S
        hM : LE.le M (nonZeroDivisors A)
        ⊢ ∀ {a b : S}, Eq (HMul.hMul a b) 0 → Or (Eq a 0) (Eq b 0)
      -/
      intro z w h
      /-
        S : Type u_2
        inst✝⁴ : CommRing S
        A : Type u_6
        inst✝³ : CommRing A
        inst✝² : IsDomain A
        inst✝¹ : Algebra A S
        M : Submonoid A
        inst✝ : IsLocalization M S
        hM : LE.le M (nonZeroDivisors A)
        z w : S
        h : Eq (HMul.hMul z w) 0
        ⊢ Or (Eq z 0) (Eq w 0)
      -/
      cases' surj M z with x hx
      /-
        case intro
        S : Type u_2
        inst✝⁴ : CommRing S
        A : Type u_6
        inst✝³ : CommRing A
        inst✝² : IsDomain A
        inst✝¹ : Algebra A S
        M : Submonoid A
        inst✝ : IsLocalization M S
        hM : LE.le M (nonZeroDivisors A)
        z w : S
        h : Eq (HMul.hMul z w) 0
        x : Prod A (Subtype fun x => Membership.mem M x)
        hx : Eq (HMul.hMul z ((algebraMap A S) ↑x.2)) ((algebraMap A S) x.1)
        ⊢ Or (Eq z 0) (Eq w 0)
      -/
      cases' surj M w with y hy
      have :
        z * w * algebraMap A S y.2 * algebraMap A S x.2 = algebraMap A S x.1 * algebraMap A S y.1 :=
        by rw [mul_assoc z, hy, ← hx]; ring
      /-
        case intro.intro
        S : Type u_2
        inst✝⁴ : CommRing S
        A : Type u_6
        inst✝³ : CommRing A
        inst✝² : IsDomain A
        inst✝¹ : Algebra A S
        M : Submonoid A
        inst✝ : IsLocalization M S
        hM : LE.le M (nonZeroDivisors A)
        z w : S
        h : Eq (HMul.hMul z w) 0
        x : Prod A (Subtype fun x => Membership.mem M x)
        hx : Eq (HMul.hMul z ((algebraMap A S) ↑x.2)) ((algebraMap A S) x.1)
        y : Prod A (Subtype fun x => Membership.mem M x)
        hy : Eq (HMul.hMul w ((algebraMap A S) ↑y.2)) ((algebraMap A S) y.1)
        this : Eq (HMul.hMul (HMul.hMul (HMul.hMul z w) ((algebraMap A S) ↑y.2)) ((alg …
        ⊢ Or (Eq z 0) (Eq w 0)
      -/
      rw [h, zero_mul, zero_mul, ← (algebraMap A S).map_mul] at this
      /-
        case intro.intro
        S : Type u_2
        inst✝⁴ : CommRing S
        A : Type u_6
        inst✝³ : CommRing A
        inst✝² : IsDomain A
        inst✝¹ : Algebra A S
        M : Submonoid A
        inst✝ : IsLocalization M S
        hM : LE.le M (nonZeroDivisors A)
        z w : S
        h : Eq (HMul.hMul z w) 0
        x : Prod A (Subtype fun x => Membership.mem M x)
        hx : Eq (HMul.hMul z ((algebraMap A S) ↑x.2)) ((algebraMap A S) x.1)
        y : Prod A (Subtype fun x => Membership.mem M x)
        hy : Eq (HMul.hMul w ((algebraMap A S) ↑y.2)) ((algebraMap A S) y.1)
        this : Eq 0 ((algebraMap A S) (HMul.hMul x.1 y.1))
        ⊢ Or (Eq z 0) (Eq w 0)
      -/
      cases' eq_zero_or_eq_zero_of_mul_eq_zero ((to_map_eq_zero_iff S hM).mp this.symm) with H H
        /-
          case intro.intro.inl
          S : Type u_2
          inst✝⁴ : CommRing S
          A : Type u_6
          inst✝³ : CommRing A
          inst✝² : IsDomain A
          inst✝¹ : Algebra A S
          M : Submonoid A
          inst✝ : IsLocalization M S
          hM : LE.le M (nonZeroDivisors A)
          z w : S
          h : Eq (HMul.hMul z w) 0
          x : Prod A (Subtype fun x => Membership.mem M x)
          hx : Eq (HMul.hMul z ((algebraMap A S) ↑x.2)) ((algebraMap A S) x.1)
          y : Prod A (Subtype fun x => Membership.mem M x)
          hy : Eq (HMul.hMul w ((algebraMap A S) ↑y.2)) ((algebraMap A S) y.1)
          this : Eq 0 ((algebraMap A S) (HMul.hMul x.1 y.1))
          H : Eq x.1 0
          ⊢ Or (Eq z 0) (Eq w 0)
        -/
      · exact Or.inl (eq_zero_of_fst_eq_zero hx H)
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.inr
          S : Type u_2
          inst✝⁴ : CommRing S
          A : Type u_6
          inst✝³ : CommRing A
          inst✝² : IsDomain A
          inst✝¹ : Algebra A S
          M : Submonoid A
          inst✝ : IsLocalization M S
          hM : LE.le M (nonZeroDivisors A)
          z w : S
          h : Eq (HMul.hMul z w) 0
          x : Prod A (Subtype fun x => Membership.mem M x)
          hx : Eq (HMul.hMul z ((algebraMap A S) ↑x.2)) ((algebraMap A S) x.1)
          y : Prod A (Subtype fun x => Membership.mem M x)
          hy : Eq (HMul.hMul w ((algebraMap A S) ↑y.2)) ((algebraMap A S) y.1)
          this : Eq 0 ((algebraMap A S) (HMul.hMul x.1 y.1))
          H : Eq y.1 0
          ⊢ Or (Eq z 0) (Eq w 0)
        -/
      · exact Or.inr (eq_zero_of_fst_eq_zero hy H) }
        /-
          🎉 no goals
        -/


/-- A `CommRing` `S` which is the localization of an integral domain `R` at a subset of
non-zero elements is an integral domain. -/
theorem isDomain_of_le_nonZeroDivisors [Algebra A S] {M : Submonoid A} [IsLocalization M S]
    (hM : M ≤ nonZeroDivisors A) : IsDomain S := by
  /-
    S : Type u_2
    inst✝⁴ : CommRing S
    A : Type u_6
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra A S
    M : Submonoid A
    inst✝ : IsLocalization M S
    hM : LE.le M (nonZeroDivisors A)
    ⊢ IsDomain S
  -/
  apply @NoZeroDivisors.to_isDomain _ _ (id _) (id _)
  · exact
      ⟨⟨(algebraMap A S) 0, (algebraMap A S) 1, fun h =>
          zero_ne_one (IsLocalization.injective S hM h)⟩⟩
    /-
      S : Type u_2
      inst✝⁴ : CommRing S
      A : Type u_6
      inst✝³ : CommRing A
      inst✝² : IsDomain A
      inst✝¹ : Algebra A S
      M : Submonoid A
      inst✝ : IsLocalization M S
      hM : LE.le M (nonZeroDivisors A)
      ⊢ NoZeroDivisors S
    -/
  · exact noZeroDivisors_of_le_nonZeroDivisors _ hM
    /-
      🎉 no goals
    -/


/-- The localization of an integral domain to a set of non-zero elements is an integral domain. -/
theorem isDomain_localization {M : Submonoid A} (hM : M ≤ nonZeroDivisors A) :
    IsDomain (Localization M) :=
  isDomain_of_le_nonZeroDivisors _ hM


