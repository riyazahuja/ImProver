/-- Two elements of a `Monoid` are `Associated` if one of them is another one
multiplied by a unit on the right. -/
def Associated [Monoid M] (x y : M) : Prop :=
  ∃ u : Mˣ, x * u = y


/-- Notation for two elements of a monoid are associated, i.e.
if one of them is another one multiplied by a unit on the right. -/
local infixl:50 " ~ᵤ " => Associated


@[refl]
protected theorem refl [Monoid M] (x : M) : x ~ᵤ x :=
         /-
           M : Type u_1
           inst✝ : Monoid M
           x : M
           ⊢ Eq (HMul.hMul x ↑1) x
         -/
  ⟨1, by simp⟩
         /-
           🎉 no goals
         -/


protected theorem rfl [Monoid M] {x : M} : x ~ᵤ x :=
  .refl x


instance [Monoid M] : IsRefl M Associated :=
  ⟨Associated.refl⟩


@[symm]
protected theorem symm [Monoid M] : ∀ {x y : M}, x ~ᵤ y → y ~ᵤ x
                               /-
                                 M : Type u_1
                                 inst✝ : Monoid M
                                 x : M
                                 u : Units M
                                 ⊢ Eq (HMul.hMul (HMul.hMul x ↑u) ↑(Inv.inv u)) x
                               -/
  | x, _, ⟨u, rfl⟩ => ⟨u⁻¹, by rw [mul_assoc, Units.mul_inv, mul_one]⟩
                               /-
                                 🎉 no goals
                               -/


instance [Monoid M] : IsSymm M Associated :=
  ⟨fun _ _ => Associated.symm⟩


protected theorem comm [Monoid M] {x y : M} : x ~ᵤ y ↔ y ~ᵤ x :=
  ⟨Associated.symm, Associated.symm⟩


@[trans]
protected theorem trans [Monoid M] : ∀ {x y z : M}, x ~ᵤ y → y ~ᵤ z → x ~ᵤ z
                                              /-
                                                M : Type u_1
                                                inst✝ : Monoid M
                                                x : M
                                                u v : Units M
                                                ⊢ Eq (HMul.hMul x ↑(HMul.hMul u v)) (HMul.hMul (HMul.hMul x ↑u) ↑v)
                                              -/
  | x, _, _, ⟨u, rfl⟩, ⟨v, rfl⟩ => ⟨u * v, by rw [Units.val_mul, mul_assoc]⟩
                                              /-
                                                🎉 no goals
                                              -/


instance [Monoid M] : IsTrans M Associated :=
  ⟨fun _ _ _ => Associated.trans⟩


/-- The setoid of the relation `x ~ᵤ y` iff there is a unit `u` such that `x * u = y` -/
protected def setoid (M : Type*) [Monoid M] :
    Setoid M where
  r := Associated
  iseqv := ⟨Associated.refl, Associated.symm, Associated.trans⟩


theorem map {M N : Type*} [Monoid M] [Monoid N] {F : Type*} [FunLike F M N] [MonoidHomClass F M N]
    (f : F) {x y : M} (ha : Associated x y) : Associated (f x) (f y) := by
  /-
    M : Type u_2
    N : Type u_3
    inst✝³ : Monoid M
    inst✝² : Monoid N
    F : Type u_4
    inst✝¹ : FunLike F M N
    inst✝ : MonoidHomClass F M N
    f : F
    x y : M
    ha : Associated x y
    ⊢ Associated (f x) (f y)
  -/
  obtain ⟨u, ha⟩ := ha
  /-
    case intro
    M : Type u_2
    N : Type u_3
    inst✝³ : Monoid M
    inst✝² : Monoid N
    F : Type u_4
    inst✝¹ : FunLike F M N
    inst✝ : MonoidHomClass F M N
    f : F
    x y : M
    u : Units M
    ha : Eq (HMul.hMul x ↑u) y
    ⊢ Associated (f x) (f y)
  -/
  exact ⟨Units.map f u, by rw [← ha, map_mul, Units.coe_map, MonoidHom.coe_coe]⟩
  /-
    🎉 no goals
  -/


theorem unit_associated_one [Monoid M] {u : Mˣ} : (u : M) ~ᵤ 1 :=
  ⟨u⁻¹, Units.mul_inv u⟩


@[simp]
theorem associated_one_iff_isUnit [Monoid M] {a : M} : (a : M) ~ᵤ 1 ↔ IsUnit a :=
  Iff.intro
    (fun h =>
      let ⟨c, h⟩ := h.symm
      h ▸ ⟨c, (one_mul _).symm⟩)
                                         /-
                                           M : Type u_1
                                           inst✝ : Monoid M
                                           a : M
                                           x✝ : IsUnit a
                                           c : Units M
                                           h : Eq (↑c) a
                                           ⊢ Eq (HMul.hMul 1 ↑c) a
                                         -/
    fun ⟨c, h⟩ => Associated.symm ⟨c, by simp [h]⟩
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem associated_zero_iff_eq_zero [MonoidWithZero M] (a : M) : a ~ᵤ 0 ↔ a = 0 :=
  Iff.intro
    (fun h => by
      /-
        M : Type u_1
        inst✝ : MonoidWithZero M
        a : M
        h : Associated a 0
        ⊢ Eq a 0
      -/
      let ⟨u, h⟩ := h.symm
      /-
        M : Type u_1
        inst✝ : MonoidWithZero M
        a : M
        h✝ : Associated a 0
        u : Units M
        h : Eq (HMul.hMul 0 ↑u) a
        ⊢ Eq a 0
      -/
      simpa using h.symm)
      /-
        🎉 no goals
      -/
    fun h => h ▸ Associated.refl a


theorem associated_one_of_mul_eq_one [CommMonoid M] {a : M} (b : M) (hab : a * b = 1) : a ~ᵤ 1 :=
  show (Units.mkOfMulEqOne a b hab : M) ~ᵤ 1 from unit_associated_one


theorem associated_one_of_associated_mul_one [CommMonoid M] {a b : M} : a * b ~ᵤ 1 → a ~ᵤ 1
                                                         /-
                                                           M : Type u_1
                                                           inst✝ : CommMonoid M
                                                           a b : M
                                                           u : Units M
                                                           h : Eq (HMul.hMul (HMul.hMul a b) ↑u) 1
                                                           ⊢ Eq (HMul.hMul a (HMul.hMul b ↑u)) 1
                                                         -/
  | ⟨u, h⟩ => associated_one_of_mul_eq_one (b * u) <| by simpa [mul_assoc] using h
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem associated_mul_unit_left {N : Type*} [Monoid N] (a u : N) (hu : IsUnit u) :
    Associated (a * u) a :=
  let ⟨u', hu⟩ := hu
  ⟨u'⁻¹, hu ▸ Units.mul_inv_cancel_right _ _⟩


theorem associated_unit_mul_left {N : Type*} [CommMonoid N] (a u : N) (hu : IsUnit u) :
    Associated (u * a) a := by
  /-
    N : Type u_2
    inst✝ : CommMonoid N
    a u : N
    hu : IsUnit u
    ⊢ Associated (HMul.hMul u a) a
  -/
  rw [mul_comm]
  /-
    N : Type u_2
    inst✝ : CommMonoid N
    a u : N
    hu : IsUnit u
    ⊢ Associated (HMul.hMul a u) a
  -/
  exact associated_mul_unit_left _ _ hu
  /-
    🎉 no goals
  -/


theorem associated_mul_unit_right {N : Type*} [Monoid N] (a u : N) (hu : IsUnit u) :
    Associated a (a * u) :=
  (associated_mul_unit_left a u hu).symm


theorem associated_unit_mul_right {N : Type*} [CommMonoid N] (a u : N) (hu : IsUnit u) :
    Associated a (u * a) :=
  (associated_unit_mul_left a u hu).symm


theorem associated_mul_isUnit_left_iff {N : Type*} [Monoid N] {a u b : N} (hu : IsUnit u) :
    Associated (a * u) b ↔ Associated a b :=
  ⟨(associated_mul_unit_right _ _ hu).trans, (associated_mul_unit_left _ _ hu).trans⟩


theorem associated_isUnit_mul_left_iff {N : Type*} [CommMonoid N] {u a b : N} (hu : IsUnit u) :
    Associated (u * a) b ↔ Associated a b := by
  /-
    N : Type u_2
    inst✝ : CommMonoid N
    u a b : N
    hu : IsUnit u
    ⊢ Iff (Associated (HMul.hMul u a) b) (Associated a b)
  -/
  rw [mul_comm]
  /-
    N : Type u_2
    inst✝ : CommMonoid N
    u a b : N
    hu : IsUnit u
    ⊢ Iff (Associated (HMul.hMul a u) b) (Associated a b)
  -/
  exact associated_mul_isUnit_left_iff hu
  /-
    🎉 no goals
  -/


theorem associated_mul_isUnit_right_iff {N : Type*} [Monoid N] {a b u : N} (hu : IsUnit u) :
    Associated a (b * u) ↔ Associated a b :=
  Associated.comm.trans <| (associated_mul_isUnit_left_iff hu).trans Associated.comm


theorem associated_isUnit_mul_right_iff {N : Type*} [CommMonoid N] {a u b : N} (hu : IsUnit u) :
    Associated a (u * b) ↔ Associated a b :=
  Associated.comm.trans <| (associated_isUnit_mul_left_iff hu).trans Associated.comm


@[simp]
theorem associated_mul_unit_left_iff {N : Type*} [Monoid N] {a b : N} {u : Units N} :
    Associated (a * u) b ↔ Associated a b :=
  associated_mul_isUnit_left_iff u.isUnit


@[simp]
theorem associated_unit_mul_left_iff {N : Type*} [CommMonoid N] {a b : N} {u : Units N} :
    Associated (↑u * a) b ↔ Associated a b :=
  associated_isUnit_mul_left_iff u.isUnit


@[simp]
theorem associated_mul_unit_right_iff {N : Type*} [Monoid N] {a b : N} {u : Units N} :
    Associated a (b * u) ↔ Associated a b :=
  associated_mul_isUnit_right_iff u.isUnit


@[simp]
theorem associated_unit_mul_right_iff {N : Type*} [CommMonoid N] {a b : N} {u : Units N} :
    Associated a (↑u * b) ↔ Associated a b :=
  associated_isUnit_mul_right_iff u.isUnit


theorem Associated.mul_left [Monoid M] (a : M) {b c : M} (h : b ~ᵤ c) : a * b ~ᵤ a * c := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a b c : M
    h : Associated b c
    ⊢ Associated (HMul.hMul a b) (HMul.hMul a c)
  -/
  obtain ⟨d, rfl⟩ := h; exact ⟨d, mul_assoc _ _ _⟩
                        /-
                          🎉 no goals
                        -/


theorem Associated.mul_right [CommMonoid M] {a b : M} (h : a ~ᵤ b) (c : M) : a * c ~ᵤ b * c := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a b : M
    h : Associated a b
    c : M
    ⊢ Associated (HMul.hMul a c) (HMul.hMul b c)
  -/
  obtain ⟨d, rfl⟩ := h; exact ⟨d, mul_right_comm _ _ _⟩
                        /-
                          🎉 no goals
                        -/


theorem Associated.mul_mul [CommMonoid M] {a₁ a₂ b₁ b₂ : M}
    (h₁ : a₁ ~ᵤ b₁) (h₂ : a₂ ~ᵤ b₂) : a₁ * a₂ ~ᵤ b₁ * b₂ := (h₁.mul_right _).trans (h₂.mul_left _)


theorem Associated.pow_pow [CommMonoid M] {a b : M} {n : ℕ} (h : a ~ᵤ b) : a ^ n ~ᵤ b ^ n := by
  induction n with
  | zero => simp [Associated.refl]
  | succ n ih => convert h.mul_mul ih <;> rw [pow_succ']


protected theorem Associated.dvd [Monoid M] {a b : M} : a ~ᵤ b → a ∣ b := fun ⟨u, hu⟩ =>
  ⟨u, hu.symm⟩


protected theorem Associated.dvd' [Monoid M] {a b : M} (h : a ~ᵤ b) : b ∣ a :=
  h.symm.dvd


protected theorem Associated.dvd_dvd [Monoid M] {a b : M} (h : a ~ᵤ b) : a ∣ b ∧ b ∣ a :=
  ⟨h.dvd, h.symm.dvd⟩


theorem associated_of_dvd_dvd [CancelMonoidWithZero M] {a b : M} (hab : a ∣ b) (hba : b ∣ a) :
    a ~ᵤ b := by
  /-
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a b : M
    hab : Dvd.dvd a b
    hba : Dvd.dvd b a
    ⊢ Associated a b
  -/
  rcases hab with ⟨c, rfl⟩
  /-
    case intro
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c : M
    hba : Dvd.dvd (HMul.hMul a c) a
    ⊢ Associated a (HMul.hMul a c)
  -/
  rcases hba with ⟨d, a_eq⟩
  /-
    case intro.intro
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c d : M
    a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
    ⊢ Associated a (HMul.hMul a c)
  -/
  by_cases ha0 : a = 0
    /-
      case pos
      M : Type u_1
      inst✝ : CancelMonoidWithZero M
      a c d : M
      a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
      ha0 : Eq a 0
      ⊢ Associated a (HMul.hMul a c)
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  have hac0 : a * c ≠ 0 := by
    intro con
    rw [con, zero_mul] at a_eq
    apply ha0 a_eq
  /-
    case neg
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c d : M
    a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
    ha0 : Not (Eq a 0)
    hac0 : Ne (HMul.hMul a c) 0
    ⊢ Associated a (HMul.hMul a c)
  -/
  have : a * (c * d) = a * 1 := by rw [← mul_assoc, ← a_eq, mul_one]
  /-
    case neg
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c d : M
    a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
    ha0 : Not (Eq a 0)
    hac0 : Ne (HMul.hMul a c) 0
    this : Eq (HMul.hMul a (HMul.hMul c d)) (HMul.hMul a 1)
    ⊢ Associated a (HMul.hMul a c)
  -/
  have hcd : c * d = 1 := mul_left_cancel₀ ha0 this
  /-
    case neg
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c d : M
    a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
    ha0 : Not (Eq a 0)
    hac0 : Ne (HMul.hMul a c) 0
    this : Eq (HMul.hMul a (HMul.hMul c d)) (HMul.hMul a 1)
    hcd : Eq (HMul.hMul c d) 1
    ⊢ Associated a (HMul.hMul a c)
  -/
  have : a * c * (d * c) = a * c * 1 := by rw [← mul_assoc, ← a_eq, mul_one]
  /-
    case neg
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c d : M
    a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
    ha0 : Not (Eq a 0)
    hac0 : Ne (HMul.hMul a c) 0
    this✝ : Eq (HMul.hMul a (HMul.hMul c d)) (HMul.hMul a 1)
    hcd : Eq (HMul.hMul c d) 1
    this : Eq (HMul.hMul (HMul.hMul a c) (HMul.hMul d c)) (HMul.hMul (HMul.hMul a  …
    ⊢ Associated a (HMul.hMul a c)
  -/
  have hdc : d * c = 1 := mul_left_cancel₀ hac0 this
  /-
    case neg
    M : Type u_1
    inst✝ : CancelMonoidWithZero M
    a c d : M
    a_eq : Eq a (HMul.hMul (HMul.hMul a c) d)
    ha0 : Not (Eq a 0)
    hac0 : Ne (HMul.hMul a c) 0
    this✝ : Eq (HMul.hMul a (HMul.hMul c d)) (HMul.hMul a 1)
    hcd : Eq (HMul.hMul c d) 1
    this : Eq (HMul.hMul (HMul.hMul a c) (HMul.hMul d c)) (HMul.hMul (HMul.hMul a  …
    hdc : Eq (HMul.hMul d c) 1
    ⊢ Associated a (HMul.hMul a c)
  -/
  exact ⟨⟨c, d, hcd, hdc⟩, rfl⟩
  /-
    🎉 no goals
  -/


theorem dvd_dvd_iff_associated [CancelMonoidWithZero M] {a b : M} : a ∣ b ∧ b ∣ a ↔ a ~ᵤ b :=
  ⟨fun ⟨h1, h2⟩ => associated_of_dvd_dvd h1 h2, Associated.dvd_dvd⟩


instance [CancelMonoidWithZero M] [DecidableRel ((· ∣ ·) : M → M → Prop)] :
    DecidableRel ((· ~ᵤ ·) : M → M → Prop) := fun _ _ => decidable_of_iff _ dvd_dvd_iff_associated


theorem Associated.dvd_iff_dvd_left [Monoid M] {a b c : M} (h : a ~ᵤ b) : a ∣ c ↔ b ∣ c :=
  let ⟨_, hu⟩ := h
  hu ▸ Units.mul_right_dvd.symm


theorem Associated.dvd_iff_dvd_right [Monoid M] {a b c : M} (h : b ~ᵤ c) : a ∣ b ↔ a ∣ c :=
  let ⟨_, hu⟩ := h
  hu ▸ Units.dvd_mul_right.symm


theorem Associated.eq_zero_iff [MonoidWithZero M] {a b : M} (h : a ~ᵤ b) : a = 0 ↔ b = 0 := by
  /-
    M : Type u_1
    inst✝ : MonoidWithZero M
    a b : M
    h : Associated a b
    ⊢ Iff (Eq a 0) (Eq b 0)
  -/
  obtain ⟨u, rfl⟩ := h
  /-
    case intro
    M : Type u_1
    inst✝ : MonoidWithZero M
    a : M
    u : Units M
    ⊢ Iff (Eq a 0) (Eq (HMul.hMul a ↑u) 0)
  -/
  rw [← Units.eq_mul_inv_iff_mul_eq, zero_mul]
  /-
    🎉 no goals
  -/


theorem Associated.ne_zero_iff [MonoidWithZero M] {a b : M} (h : a ~ᵤ b) : a ≠ 0 ↔ b ≠ 0 :=
  not_congr h.eq_zero_iff


theorem Associated.neg_left [Monoid M] [HasDistribNeg M] {a b : M} (h : Associated a b) :
    Associated (-a) b :=
                            /-
                              M : Type u_1
                              inst✝¹ : Monoid M
                              inst✝ : HasDistribNeg M
                              a b : M
                              h : Associated a b
                              u : Units M
                              hu : Eq (HMul.hMul a ↑u) b
                              ⊢ Eq (HMul.hMul (Neg.neg a) ↑(Neg.neg u)) b
                            -/
  let ⟨u, hu⟩ := h; ⟨-u, by simp [hu]⟩
                            /-
                              🎉 no goals
                            -/


theorem Associated.neg_right [Monoid M] [HasDistribNeg M] {a b : M} (h : Associated a b) :
    Associated a (-b) :=
  h.symm.neg_left.symm


theorem Associated.neg_neg [Monoid M] [HasDistribNeg M] {a b : M} (h : Associated a b) :
    Associated (-a) (-b) :=
  h.neg_left.neg_right


protected theorem Associated.prime [CommMonoidWithZero M] {p q : M} (h : p ~ᵤ q) (hp : Prime p) :
    Prime q :=
  ⟨h.ne_zero_iff.1 hp.ne_zero,
    let ⟨u, hu⟩ := h
                                             /-
                                               M : Type u_1
                                               inst✝ : CommMonoidWithZero M
                                               p q : M
                                               h : Associated p q
                                               hp : Prime p
                                               u : Units M
                                               hu : Eq (HMul.hMul p ↑u) q
                                               x✝ : IsUnit q
                                               v : Units M
                                               hv : Eq (↑v) q
                                               ⊢ Eq (↑(HMul.hMul v (Inv.inv u))) p
                                             -/
    ⟨fun ⟨v, hv⟩ => hp.not_unit ⟨v * u⁻¹, by simp [hv, hu.symm]⟩,
                                             /-
                                               🎉 no goals
                                             -/
      hu ▸ by
        /-
          M : Type u_1
          inst✝ : CommMonoidWithZero M
          p q : M
          h : Associated p q
          hp : Prime p
          u : Units M
          hu : Eq (HMul.hMul p ↑u) q
          ⊢ ∀ (a b : M), Dvd.dvd (HMul.hMul p ↑u) (HMul.hMul a b) → Or (Dvd.dvd (HMul.hM …
        -/
        simp only [IsUnit.mul_iff, Units.isUnit, and_true, IsUnit.mul_right_dvd]
        /-
          M : Type u_1
          inst✝ : CommMonoidWithZero M
          p q : M
          h : Associated p q
          hp : Prime p
          u : Units M
          hu : Eq (HMul.hMul p ↑u) q
          ⊢ ∀ (a b : M), Dvd.dvd p (HMul.hMul a b) → Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        intro a b
        /-
          M : Type u_1
          inst✝ : CommMonoidWithZero M
          p q : M
          h : Associated p q
          hp : Prime p
          u : Units M
          hu : Eq (HMul.hMul p ↑u) q
          a b : M
          ⊢ Dvd.dvd p (HMul.hMul a b) → Or (Dvd.dvd p a) (Dvd.dvd p b)
        -/
        exact hp.dvd_or_dvd⟩⟩
        /-
          🎉 no goals
        -/


theorem prime_mul_iff [CancelCommMonoidWithZero M] {x y : M} :
    Prime (x * y) ↔ (Prime x ∧ IsUnit y) ∨ (IsUnit x ∧ Prime y) := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    x y : M
    ⊢ Iff (Prime (HMul.hMul x y)) (Or (And (Prime x) (IsUnit y)) (And (IsUnit x) ( …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      x y : M
      h : Prime (HMul.hMul x y)
      ⊢ Or (And (Prime x) (IsUnit y)) (And (IsUnit x) (Prime y))
    -/
  · rcases of_irreducible_mul h.irreducible with hx | hy
      /-
        case refine_1.inl
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        x y : M
        h : Prime (HMul.hMul x y)
        hx : IsUnit x
        ⊢ Or (And (Prime x) (IsUnit y)) (And (IsUnit x) (Prime y))
      -/
    · exact Or.inr ⟨hx, (associated_unit_mul_left y x hx).prime h⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        x y : M
        h : Prime (HMul.hMul x y)
        hy : IsUnit y
        ⊢ Or (And (Prime x) (IsUnit y)) (And (IsUnit x) (Prime y))
      -/
    · exact Or.inl ⟨(associated_mul_unit_left x y hy).prime h, hy⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      x y : M
      ⊢ Or (And (Prime x) (IsUnit y)) (And (IsUnit x) (Prime y)) → Prime (HMul.hMul  …
    -/
  · rintro (⟨hx, hy⟩ | ⟨hx, hy⟩)
      /-
        case refine_2.inl.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        x y : M
        hx : Prime x
        hy : IsUnit y
        ⊢ Prime (HMul.hMul x y)
      -/
    · exact (associated_mul_unit_left x y hy).symm.prime hx
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        x y : M
        hx : IsUnit x
        hy : Prime y
        ⊢ Prime (HMul.hMul x y)
      -/
    · exact (associated_unit_mul_right y x hx).prime hy
      /-
        🎉 no goals
      -/


@[simp]
lemma prime_pow_iff [CancelCommMonoidWithZero M] {p : M} {n : ℕ} :
    Prime (p ^ n) ↔ Prime p ∧ n = 1 := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    n : Nat
    ⊢ Iff (Prime (HPow.hPow p n)) (And (Prime p) (Eq n 1))
  -/
  refine ⟨fun hp ↦ ?_, fun ⟨hp, hn⟩ ↦ by simpa [hn]⟩
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    n : Nat
    hp : Prime (HPow.hPow p n)
    ⊢ And (Prime p) (Eq n 1)
  -/
  suffices n = 1 by aesop
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    n : Nat
    hp : Prime (HPow.hPow p n)
    ⊢ Eq n 1
  -/
  rcases n with - | n
    /-
      case zero
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      hp : Prime (HPow.hPow p 0)
      ⊢ Eq 0 1
    -/
  · simp at hp
    /-
      🎉 no goals
    -/
    /-
      case succ
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      n : Nat
      hp : Prime (HPow.hPow p (HAdd.hAdd n 1))
      ⊢ Eq (HAdd.hAdd n 1) 1
    -/
  · rw [Nat.succ.injEq]
    /-
      case succ
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      n : Nat
      hp : Prime (HPow.hPow p (HAdd.hAdd n 1))
      ⊢ Eq n 0
    -/
    rw [pow_succ', prime_mul_iff] at hp
    /-
      case succ
      M : Type u_1
      inst✝ : CancelCommMonoidWithZero M
      p : M
      n : Nat
      hp : Or (And (Prime p) (IsUnit (HPow.hPow p n))) (And (IsUnit p) (Prime (HPow. …
      ⊢ Eq n 0
    -/
    rcases hp with ⟨hp, hpn⟩ | ⟨hp, hpn⟩
      /-
        case succ.inl.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        p : M
        n : Nat
        hp : Prime p
        hpn : IsUnit (HPow.hPow p n)
        ⊢ Eq n 0
      -/
    · by_contra contra
      /-
        case succ.inl.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        p : M
        n : Nat
        hp : Prime p
        hpn : IsUnit (HPow.hPow p n)
        contra : Not (Eq n 0)
        ⊢ False
      -/
      rw [isUnit_pow_iff contra] at hpn
      /-
        case succ.inl.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        p : M
        n : Nat
        hp : Prime p
        hpn : IsUnit p
        contra : Not (Eq n 0)
        ⊢ False
      -/
      exact hp.not_unit hpn
      /-
        🎉 no goals
      -/
      /-
        case succ.inr.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        p : M
        n : Nat
        hp : IsUnit p
        hpn : Prime (HPow.hPow p n)
        ⊢ Eq n 0
      -/
    · exfalso
      /-
        case succ.inr.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        p : M
        n : Nat
        hp : IsUnit p
        hpn : Prime (HPow.hPow p n)
        ⊢ False
      -/
      exact hpn.not_unit (hp.pow n)
      /-
        🎉 no goals
      -/


theorem Irreducible.dvd_iff [Monoid M] {x y : M} (hx : Irreducible x) :
    y ∣ x ↔ IsUnit y ∨ Associated x y := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    x y : M
    hx : Irreducible x
    ⊢ Iff (Dvd.dvd y x) (Or (IsUnit y) (Associated x y))
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : Monoid M
      x y : M
      hx : Irreducible x
      ⊢ Dvd.dvd y x → Or (IsUnit y) (Associated x y)
    -/
  · rintro ⟨z, hz⟩
    /-
      case mp.intro
      M : Type u_1
      inst✝ : Monoid M
      x y : M
      hx : Irreducible x
      z : M
      hz : Eq x (HMul.hMul y z)
      ⊢ Or (IsUnit y) (Associated x y)
    -/
    obtain (h|h) := hx.isUnit_or_isUnit hz
      /-
        case mp.intro.inl
        M : Type u_1
        inst✝ : Monoid M
        x y : M
        hx : Irreducible x
        z : M
        hz : Eq x (HMul.hMul y z)
        h : IsUnit y
        ⊢ Or (IsUnit y) (Associated x y)
      -/
    · exact Or.inl h
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.inr
        M : Type u_1
        inst✝ : Monoid M
        x y : M
        hx : Irreducible x
        z : M
        hz : Eq x (HMul.hMul y z)
        h : IsUnit z
        ⊢ Or (IsUnit y) (Associated x y)
      -/
    · rw [hz]
      /-
        case mp.intro.inr
        M : Type u_1
        inst✝ : Monoid M
        x y : M
        hx : Irreducible x
        z : M
        hz : Eq x (HMul.hMul y z)
        h : IsUnit z
        ⊢ Or (IsUnit y) (Associated (HMul.hMul y z) y)
      -/
      exact Or.inr (associated_mul_unit_left _ _ h)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      M : Type u_1
      inst✝ : Monoid M
      x y : M
      hx : Irreducible x
      ⊢ Or (IsUnit y) (Associated x y) → Dvd.dvd y x
    -/
  · rintro (hy|h)
      /-
        case mpr.inl
        M : Type u_1
        inst✝ : Monoid M
        x y : M
        hx : Irreducible x
        hy : IsUnit y
        ⊢ Dvd.dvd y x
      -/
    · exact hy.dvd
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        M : Type u_1
        inst✝ : Monoid M
        x y : M
        hx : Irreducible x
        h : Associated x y
        ⊢ Dvd.dvd y x
      -/
    · exact h.symm.dvd
      /-
        🎉 no goals
      -/


theorem Irreducible.associated_of_dvd [Monoid M] {p q : M} (p_irr : Irreducible p)
    (q_irr : Irreducible q) (dvd : p ∣ q) : Associated p q :=
  ((q_irr.dvd_iff.mp dvd).resolve_left p_irr.not_unit).symm


theorem Irreducible.dvd_irreducible_iff_associated [Monoid M] {p q : M}
    (pp : Irreducible p) (qp : Irreducible q) : p ∣ q ↔ Associated p q :=
  ⟨Irreducible.associated_of_dvd pp qp, Associated.dvd⟩


theorem Prime.associated_of_dvd [CancelCommMonoidWithZero M] {p q : M} (p_prime : Prime p)
    (q_prime : Prime q) (dvd : p ∣ q) : Associated p q :=
  p_prime.irreducible.associated_of_dvd q_prime.irreducible dvd


theorem Prime.dvd_prime_iff_associated [CancelCommMonoidWithZero M] {p q : M} (pp : Prime p)
    (qp : Prime q) : p ∣ q ↔ Associated p q :=
  pp.irreducible.dvd_irreducible_iff_associated qp.irreducible


theorem Associated.prime_iff [CommMonoidWithZero M] {p q : M} (h : p ~ᵤ q) : Prime p ↔ Prime q :=
  ⟨h.prime, h.symm.prime⟩


protected theorem Associated.isUnit [Monoid M] {a b : M} (h : a ~ᵤ b) : IsUnit a → IsUnit b :=
  let ⟨u, hu⟩ := h
                            /-
                              M : Type u_1
                              inst✝ : Monoid M
                              a b : M
                              h : Associated a b
                              u : Units M
                              hu : Eq (HMul.hMul a ↑u) b
                              x✝ : IsUnit a
                              v : Units M
                              hv : Eq (↑v) a
                              ⊢ Eq (↑(HMul.hMul v u)) b
                            -/
  fun ⟨v, hv⟩ => ⟨v * u, by simp [hv, hu.symm]⟩
                            /-
                              🎉 no goals
                            -/


theorem Associated.isUnit_iff [Monoid M] {a b : M} (h : a ~ᵤ b) : IsUnit a ↔ IsUnit b :=
  ⟨h.isUnit, h.symm.isUnit⟩


theorem Irreducible.isUnit_iff_not_associated_of_dvd [Monoid M]
    {x y : M} (hx : Irreducible x) (hy : y ∣ x) : IsUnit y ↔ ¬ Associated x y :=
  ⟨fun hy hxy => hx.1 (hxy.symm.isUnit hy), (hx.dvd_iff.mp hy).resolve_right⟩


protected theorem Associated.irreducible [Monoid M] {p q : M} (h : p ~ᵤ q) (hp : Irreducible p) :
    Irreducible q :=
  ⟨mt h.symm.isUnit hp.1,
    let ⟨u, hu⟩ := h
    fun a b hab =>
    have hpab : p = a * (b * (u⁻¹ : Mˣ)) :=
      calc
                                     /-
                                       M : Type u_1
                                       inst✝ : Monoid M
                                       p q : M
                                       h : Associated p q
                                       hp : Irreducible p
                                       u : Units M
                                       hu : Eq (HMul.hMul p ↑u) q
                                       a b : M
                                       hab : Eq q (HMul.hMul a b)
                                       ⊢ Eq p (HMul.hMul (HMul.hMul p ↑u) ↑(Inv.inv u))
                                     -/
        p = p * u * (u⁻¹ : Mˣ) := by simp
                                     /-
                                       🎉 no goals
                                     -/
                    /-
                      M : Type u_1
                      inst✝ : Monoid M
                      p q : M
                      h : Associated p q
                      hp : Irreducible p
                      u : Units M
                      hu : Eq (HMul.hMul p ↑u) q
                      a b : M
                      hab : Eq q (HMul.hMul a b)
                      ⊢ Eq (HMul.hMul (HMul.hMul p ↑u) ↑(Inv.inv u)) (HMul.hMul a (HMul.hMul b ↑(Inv …
                    -/
        _ = _ := by rw [hu]; simp [hab, mul_assoc]
                             /-
                               🎉 no goals
                             -/

                                                                            /-
                                                                              M : Type u_1
                                                                              inst✝ : Monoid M
                                                                              p q : M
                                                                              h : Associated p q
                                                                              hp : Irreducible p
                                                                              u : Units M
                                                                              hu : Eq (HMul.hMul p ↑u) q
                                                                              a b : M
                                                                              hab : Eq q (HMul.hMul a b)
                                                                              hpab : Eq p (HMul.hMul a (HMul.hMul b ↑(Inv.inv u)))
                                                                              x✝ : IsUnit (HMul.hMul b ↑(Inv.inv u))
                                                                              v : Units M
                                                                              hv : Eq (↑v) (HMul.hMul b ↑(Inv.inv u))
                                                                              ⊢ Eq (↑(HMul.hMul v u)) b
                                                                            -/
    (hp.isUnit_or_isUnit hpab).elim Or.inl fun ⟨v, hv⟩ => Or.inr ⟨v * u, by simp [hv]⟩⟩
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


protected theorem Associated.irreducible_iff [Monoid M] {p q : M} (h : p ~ᵤ q) :
    Irreducible p ↔ Irreducible q :=
  ⟨h.irreducible, h.symm.irreducible⟩


theorem Associated.of_mul_left [CancelCommMonoidWithZero M] {a b c d : M} (h : a * b ~ᵤ c * d)
    (h₁ : a ~ᵤ c) (ha : a ≠ 0) : b ~ᵤ d :=
  let ⟨u, hu⟩ := h
  let ⟨v, hv⟩ := Associated.symm h₁
  ⟨u * (v : Mˣ),
    mul_left_cancel₀ ha
      (by
        /-
          M : Type u_1
          inst✝ : CancelCommMonoidWithZero M
          a b c d : M
          h : Associated (HMul.hMul a b) (HMul.hMul c d)
          h₁ : Associated a c
          ha : Ne a 0
          u : Units M
          hu : Eq (HMul.hMul (HMul.hMul a b) ↑u) (HMul.hMul c d)
          v : Units M
          hv : Eq (HMul.hMul c ↑v) a
          ⊢ Eq (HMul.hMul a (HMul.hMul b ↑(HMul.hMul u v))) (HMul.hMul a d)
        -/
        rw [← hv, mul_assoc c (v : M) d, mul_left_comm c, ← hu]
        /-
          M : Type u_1
          inst✝ : CancelCommMonoidWithZero M
          a b c d : M
          h : Associated (HMul.hMul a b) (HMul.hMul c d)
          h₁ : Associated a c
          ha : Ne a 0
          u : Units M
          hu : Eq (HMul.hMul (HMul.hMul a b) ↑u) (HMul.hMul c d)
          v : Units M
          hv : Eq (HMul.hMul c ↑v) a
          ⊢ Eq (HMul.hMul (HMul.hMul c ↑v) (HMul.hMul b ↑(HMul.hMul u v))) (HMul.hMul (↑ …
        -/
        simp [hv.symm, mul_assoc, mul_comm, mul_left_comm])⟩
        /-
          🎉 no goals
        -/


theorem Associated.of_mul_right [CancelCommMonoidWithZero M] {a b c d : M} :
    a * b ~ᵤ c * d → b ~ᵤ d → b ≠ 0 → a ~ᵤ c := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    a b c d : M
    ⊢ Associated (HMul.hMul a b) (HMul.hMul c d) → Associated b d → Ne b 0 → Assoc …
  -/
  rw [mul_comm a, mul_comm c]; exact Associated.of_mul_left
                               /-
                                 🎉 no goals
                               -/


theorem Associated.of_pow_associated_of_prime [CancelCommMonoidWithZero M] {p₁ p₂ : M} {k₁ k₂ : ℕ}
    (hp₁ : Prime p₁) (hp₂ : Prime p₂) (hk₁ : 0 < k₁) (h : p₁ ^ k₁ ~ᵤ p₂ ^ k₂) : p₁ ~ᵤ p₂ := by
  have : p₁ ∣ p₂ ^ k₂ := by
    rw [← h.dvd_iff_dvd_right]
    apply dvd_pow_self _ hk₁.ne'
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p₁ p₂ : M
    k₁ k₂ : Nat
    hp₁ : Prime p₁
    hp₂ : Prime p₂
    hk₁ : LT.lt 0 k₁
    h : Associated (HPow.hPow p₁ k₁) (HPow.hPow p₂ k₂)
    this : Dvd.dvd p₁ (HPow.hPow p₂ k₂)
    ⊢ Associated p₁ p₂
  -/
  rw [← hp₁.dvd_prime_iff_associated hp₂]
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p₁ p₂ : M
    k₁ k₂ : Nat
    hp₁ : Prime p₁
    hp₂ : Prime p₂
    hk₁ : LT.lt 0 k₁
    h : Associated (HPow.hPow p₁ k₁) (HPow.hPow p₂ k₂)
    this : Dvd.dvd p₁ (HPow.hPow p₂ k₂)
    ⊢ Dvd.dvd p₁ p₂
  -/
  exact hp₁.dvd_of_dvd_pow this
  /-
    🎉 no goals
  -/


theorem Associated.of_pow_associated_of_prime' [CancelCommMonoidWithZero M] {p₁ p₂ : M} {k₁ k₂ : ℕ}
    (hp₁ : Prime p₁) (hp₂ : Prime p₂) (hk₂ : 0 < k₂) (h : p₁ ^ k₁ ~ᵤ p₂ ^ k₂) : p₁ ~ᵤ p₂ :=
  (h.symm.of_pow_associated_of_prime hp₂ hp₁ hk₂).symm


/-- See also `Irreducible.coprime_iff_not_dvd`. -/
lemma Irreducible.isRelPrime_iff_not_dvd [Monoid M] {p n : M} (hp : Irreducible p) :
    IsRelPrime p n ↔ ¬ p ∣ n := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    p n : M
    hp : Irreducible p
    ⊢ Iff (IsRelPrime p n) (Not (Dvd.dvd p n))
  -/
  refine ⟨fun h contra ↦ hp.not_unit (h dvd_rfl contra), fun hpn d hdp hdn ↦ ?_⟩
  /-
    M : Type u_1
    inst✝ : Monoid M
    p n : M
    hp : Irreducible p
    hpn : Not (Dvd.dvd p n)
    d : M
    hdp : Dvd.dvd d p
    hdn : Dvd.dvd d n
    ⊢ IsUnit d
  -/
  contrapose! hpn
  /-
    M : Type u_1
    inst✝ : Monoid M
    p n : M
    hp : Irreducible p
    d : M
    hdp : Dvd.dvd d p
    hdn : Dvd.dvd d n
    hpn : Not (IsUnit d)
    ⊢ Dvd.dvd p n
  -/
  suffices Associated p d from this.dvd.trans hdn
  /-
    M : Type u_1
    inst✝ : Monoid M
    p n : M
    hp : Irreducible p
    d : M
    hdp : Dvd.dvd d p
    hdn : Dvd.dvd d n
    hpn : Not (IsUnit d)
    ⊢ Associated p d
  -/
  exact (hp.dvd_iff.mp hdp).resolve_left hpn
  /-
    🎉 no goals
  -/


lemma Irreducible.dvd_or_isRelPrime [Monoid M] {p n : M} (hp : Irreducible p) :
    p ∣ n ∨ IsRelPrime p n := Classical.or_iff_not_imp_left.mpr hp.isRelPrime_iff_not_dvd.2


theorem associated_iff_eq {x y : M} : x ~ᵤ y ↔ x = y := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : Subsingleton (Units M)
    x y : M
    ⊢ Iff (Associated x y) (Eq x y)
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝¹ : Monoid M
      inst✝ : Subsingleton (Units M)
      x y : M
      ⊢ Associated x y → Eq x y
    -/
  · rintro ⟨c, rfl⟩
    /-
      case mp.intro
      M : Type u_1
      inst✝¹ : Monoid M
      inst✝ : Subsingleton (Units M)
      x : M
      c : Units M
      ⊢ Eq x (HMul.hMul x ↑c)
    -/
    rw [units_eq_one c, Units.val_one, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      M : Type u_1
      inst✝¹ : Monoid M
      inst✝ : Subsingleton (Units M)
      x y : M
      ⊢ Eq x y → Associated x y
    -/
  · rintro rfl
    /-
      case mpr
      M : Type u_1
      inst✝¹ : Monoid M
      inst✝ : Subsingleton (Units M)
      x : M
      ⊢ Associated x x
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem associated_eq_eq : (Associated : M → M → Prop) = Eq := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : Subsingleton (Units M)
    ⊢ Eq Associated Eq
  -/
  ext
  /-
    case h.h.a
    M : Type u_1
    inst✝¹ : Monoid M
    inst✝ : Subsingleton (Units M)
    x✝¹ x✝ : M
    ⊢ Iff (Associated x✝¹ x✝) (Eq x✝¹ x✝)
  -/
  rw [associated_iff_eq]
  /-
    🎉 no goals
  -/


theorem prime_dvd_prime_iff_eq {M : Type*} [CancelCommMonoidWithZero M] [Subsingleton Mˣ] {p q : M}
    (pp : Prime p) (qp : Prime q) : p ∣ q ↔ p = q := by
  /-
    M : Type u_2
    inst✝¹ : CancelCommMonoidWithZero M
    inst✝ : Subsingleton (Units M)
    p q : M
    pp : Prime p
    qp : Prime q
    ⊢ Iff (Dvd.dvd p q) (Eq p q)
  -/
  rw [pp.dvd_prime_iff_associated qp, ← associated_eq_eq]
  /-
    🎉 no goals
  -/


theorem eq_of_prime_pow_eq (hp₁ : Prime p₁) (hp₂ : Prime p₂) (hk₁ : 0 < k₁)
    (h : p₁ ^ k₁ = p₂ ^ k₂) : p₁ = p₂ := by
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : Subsingleton (Units R)
    p₁ p₂ : R
    k₁ k₂ : Nat
    hp₁ : Prime p₁
    hp₂ : Prime p₂
    hk₁ : LT.lt 0 k₁
    h : Eq (HPow.hPow p₁ k₁) (HPow.hPow p₂ k₂)
    ⊢ Eq p₁ p₂
  -/
  rw [← associated_iff_eq] at h ⊢
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : Subsingleton (Units R)
    p₁ p₂ : R
    k₁ k₂ : Nat
    hp₁ : Prime p₁
    hp₂ : Prime p₂
    hk₁ : LT.lt 0 k₁
    h : Associated (HPow.hPow p₁ k₁) (HPow.hPow p₂ k₂)
    ⊢ Associated p₁ p₂
  -/
  apply h.of_pow_associated_of_prime hp₁ hp₂ hk₁
  /-
    🎉 no goals
  -/


theorem eq_of_prime_pow_eq' (hp₁ : Prime p₁) (hp₂ : Prime p₂) (hk₁ : 0 < k₂)
    (h : p₁ ^ k₁ = p₂ ^ k₂) : p₁ = p₂ := by
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : Subsingleton (Units R)
    p₁ p₂ : R
    k₁ k₂ : Nat
    hp₁ : Prime p₁
    hp₂ : Prime p₂
    hk₁ : LT.lt 0 k₂
    h : Eq (HPow.hPow p₁ k₁) (HPow.hPow p₂ k₂)
    ⊢ Eq p₁ p₂
  -/
  rw [← associated_iff_eq] at h ⊢
  /-
    R : Type u_2
    inst✝¹ : CancelCommMonoidWithZero R
    inst✝ : Subsingleton (Units R)
    p₁ p₂ : R
    k₁ k₂ : Nat
    hp₁ : Prime p₁
    hp₂ : Prime p₂
    hk₁ : LT.lt 0 k₂
    h : Associated (HPow.hPow p₁ k₁) (HPow.hPow p₂ k₂)
    ⊢ Associated p₁ p₂
  -/
  apply h.of_pow_associated_of_prime' hp₁ hp₂ hk₁
  /-
    🎉 no goals
  -/


/-- The quotient of a monoid by the `Associated` relation. Two elements `x` and `y`
  are associated iff there is a unit `u` such that `x * u = y`. There is a natural
  monoid structure on `Associates M`. -/
abbrev Associates (M : Type*) [Monoid M] : Type _ :=
  Quotient (Associated.setoid M)


/-- The canonical quotient map from a monoid `M` into the `Associates` of `M` -/
protected abbrev mk {M : Type*} [Monoid M] (a : M) : Associates M :=
  ⟦a⟧


instance [Monoid M] : Inhabited (Associates M) :=
  ⟨⟦1⟧⟩


theorem mk_eq_mk_iff_associated [Monoid M] {a b : M} : Associates.mk a = Associates.mk b ↔ a ~ᵤ b :=
  Iff.intro Quotient.exact Quot.sound


theorem quotient_mk_eq_mk [Monoid M] (a : M) : ⟦a⟧ = Associates.mk a :=
  rfl


theorem quot_mk_eq_mk [Monoid M] (a : M) : Quot.mk Setoid.r a = Associates.mk a :=
  rfl


@[simp]
theorem quot_out [Monoid M] (a : Associates M) : Associates.mk (Quot.out a) = a := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Associates M
    ⊢ Eq (Associates.mk (Quot.out a)) a
  -/
  rw [← quot_mk_eq_mk, Quot.out_eq]
  /-
    🎉 no goals
  -/


theorem mk_quot_out [Monoid M] (a : M) : Quot.out (Associates.mk a) ~ᵤ a := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : M
    ⊢ Associated (Quot.out (Associates.mk a)) a
  -/
  rw [← Associates.mk_eq_mk_iff_associated, Associates.quot_out]
  /-
    🎉 no goals
  -/


theorem forall_associated [Monoid M] {p : Associates M → Prop} :
    (∀ a, p a) ↔ ∀ a, p (Associates.mk a) :=
  Iff.intro (fun h _ => h _) fun h a => Quotient.inductionOn a h


theorem mk_surjective [Monoid M] : Function.Surjective (@Associates.mk M _) :=
  forall_associated.2 fun a => ⟨a, rfl⟩


instance [Monoid M] : One (Associates M) :=
  ⟨⟦1⟧⟩


@[simp]
theorem mk_one [Monoid M] : Associates.mk (1 : M) = 1 :=
  rfl


theorem one_eq_mk_one [Monoid M] : (1 : Associates M) = Associates.mk 1 :=
  rfl


@[simp]
theorem mk_eq_one [Monoid M] {a : M} : Associates.mk a = 1 ↔ IsUnit a := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : M
    ⊢ Iff (Eq (Associates.mk a) 1) (IsUnit a)
  -/
  rw [← mk_one, mk_eq_mk_iff_associated, associated_one_iff_isUnit]
  /-
    🎉 no goals
  -/


instance [Monoid M] : Bot (Associates M) :=
  ⟨1⟩


theorem bot_eq_one [Monoid M] : (⊥ : Associates M) = 1 :=
  rfl


theorem exists_rep [Monoid M] (a : Associates M) : ∃ a0 : M, Associates.mk a0 = a :=
  Quot.exists_rep a


instance [Monoid M] [Subsingleton M] :
    Unique (Associates M) where
  default := 1
  uniq := forall_associated.2 fun _ ↦ mk_eq_one.2 <| isUnit_of_subsingleton _


theorem mk_injective [Monoid M] [Subsingleton Mˣ] : Function.Injective (@Associates.mk M _) :=
  fun _ _ h => associated_iff_eq.mp (Associates.mk_eq_mk_iff_associated.mp h)


instance instMul : Mul (Associates M) :=
  ⟨Quotient.map₂ (· * ·) fun _ _ h₁ _ _ h₂ ↦ h₁.mul_mul h₂⟩


theorem mk_mul_mk {x y : M} : Associates.mk x * Associates.mk y = Associates.mk (x * y) :=
  rfl


instance instCommMonoid : CommMonoid (Associates M) where
  one := 1
  mul := (· * ·)
                                                                       /-
                                                                         M : Type u_1
                                                                         inst✝ : CommMonoid M
                                                                         a' : Associates M
                                                                         a : M
                                                                         ⊢ Eq (Quotient.mk (Associated.setoid M) (HMul.hMul a 1)) (Quotient.mk (Associa …
                                                                       -/
                                                                       /-
                                                                         M : Type u_1
                                                                         inst✝ : CommMonoid M
                                                                         a' : Associates M
                                                                         a : M
                                                                         ⊢ Eq (Quotient.mk (Associated.setoid M) (HMul.hMul 1 a)) (Quotient.mk (Associa …
                                                                       -/
  mul_one a' := Quotient.inductionOn a' fun a => show ⟦a * 1⟧ = ⟦a⟧ by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                          /-
                                            M : Type u_1
                                            inst✝ : CommMonoid M
                                            a' b' c' : Associates M
                                            a b c : M
                                            ⊢ Eq (Quotient.mk (Associated.setoid M) (HMul.hMul (HMul.hMul a b) c)) (Quotie …
                                          -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                          /-
                                            🎉 no goals
                                          -/
  one_mul a' := Quotient.inductionOn a' fun a => show ⟦1 * a⟧ = ⟦a⟧ by simp
  mul_assoc a' b' c' :=
    Quotient.inductionOn₃ a' b' c' fun a b c =>
      show ⟦a * b * c⟧ = ⟦a * (b * c)⟧ by rw [mul_assoc]
  mul_comm a' b' :=
                                                                     /-
                                                                       M : Type u_1
                                                                       inst✝ : CommMonoid M
                                                                       a' b' : Associates M
                                                                       a b : M
                                                                       ⊢ Eq (Quotient.mk (Associated.setoid M) (HMul.hMul a b)) (Quotient.mk (Associa …
                                                                     -/
    Quotient.inductionOn₂ a' b' fun a b => show ⟦a * b⟧ = ⟦b * a⟧ by rw [mul_comm]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance instPreorder : Preorder (Associates M) where
  le := Dvd.dvd
  le_refl := dvd_refl
  le_trans _ _ _ := dvd_trans


/-- `Associates.mk` as a `MonoidHom`. -/
protected def mkMonoidHom : M →* Associates M where
  toFun := Associates.mk
  map_one' := mk_one
  map_mul' _ _ := mk_mul_mk


@[simp]
theorem mkMonoidHom_apply (a : M) : Associates.mkMonoidHom a = Associates.mk a :=
  rfl


theorem associated_map_mk {f : Associates M →* M} (hinv : Function.RightInverse f Associates.mk)
    (a : M) : a ~ᵤ f (Associates.mk a) :=
  Associates.mk_eq_mk_iff_associated.1 (hinv (Associates.mk a)).symm


theorem mk_pow (a : M) (n : ℕ) : Associates.mk (a ^ n) = Associates.mk a ^ n := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : M
    n : Nat
    ⊢ Eq (Associates.mk (HPow.hPow a n)) (HPow.hPow (Associates.mk a) n)
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, pow_succ, Associates.mk_mul_mk.symm]
                  /-
                    🎉 no goals
                  -/


theorem dvd_eq_le : ((· ∣ ·) : Associates M → Associates M → Prop) = (· ≤ ·) :=
  rfl


instance uniqueUnits : Unique (Associates M)ˣ where
  uniq := by
    /-
      M : Type u_1
      inst✝ : CommMonoid M
      ⊢ ∀ (a : Units (Associates M)), Eq a Inhabited.default
    -/
    rintro ⟨a, b, hab, hba⟩
    /-
      case mk
      M : Type u_1
      inst✝ : CommMonoid M
      a b : Associates M
      hab : Eq (HMul.hMul a b) 1
      hba : Eq (HMul.hMul b a) 1
      ⊢ Eq { val := a, inv := b, val_inv := hab, inv_val := hba } Inhabited.default
    -/
    revert hab hba
    exact Quotient.inductionOn₂ a b <| fun a b hab hba ↦ Units.ext <| Quotient.sound <|
      associated_one_of_associated_mul_one <| Quotient.exact hab


@[deprecated (since := "2024-07-22")] alias mul_eq_one_iff := mul_eq_one

@[deprecated (since := "2024-07-22")] protected alias units_eq_one := Subsingleton.elim


@[simp]
theorem coe_unit_eq_one (u : (Associates M)ˣ) : (u : Associates M) = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    u : Units (Associates M)
    ⊢ Eq (↑u) 1
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


theorem isUnit_iff_eq_one (a : Associates M) : IsUnit a ↔ a = 1 :=
  Iff.intro (fun ⟨_, h⟩ => h ▸ coe_unit_eq_one _) fun h => h.symm ▸ isUnit_one


theorem isUnit_iff_eq_bot {a : Associates M} : IsUnit a ↔ a = ⊥ := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : Associates M
    ⊢ Iff (IsUnit a) (Eq a Bot.bot)
  -/
  rw [Associates.isUnit_iff_eq_one, bot_eq_one]
  /-
    🎉 no goals
  -/


theorem isUnit_mk {a : M} : IsUnit (Associates.mk a) ↔ IsUnit a :=
  calc
    IsUnit (Associates.mk a) ↔ a ~ᵤ 1 := by
      /-
        M : Type u_1
        inst✝ : CommMonoid M
        a : M
        ⊢ Iff (IsUnit (Associates.mk a)) (Associated a 1)
      -/
      rw [isUnit_iff_eq_one, one_eq_mk_one, mk_eq_mk_iff_associated]
      /-
        🎉 no goals
      -/
    _ ↔ IsUnit a := associated_one_iff_isUnit


theorem mul_mono {a b c d : Associates M} (h₁ : a ≤ b) (h₂ : c ≤ d) : a * c ≤ b * d :=
  let ⟨x, hx⟩ := h₁
  let ⟨y, hy⟩ := h₂
             /-
               M : Type u_1
               inst✝ : CommMonoid M
               a b c d : Associates M
               h₁ : LE.le a b
               h₂ : LE.le c d
               x : Associates M
               hx : Eq b (HMul.hMul a x)
               y : Associates M
               hy : Eq d (HMul.hMul c y)
               ⊢ Eq (HMul.hMul b d) (HMul.hMul (HMul.hMul a c) (HMul.hMul x y))
             -/
  ⟨x * y, by simp [hx, hy, mul_comm, mul_assoc, mul_left_comm]⟩
             /-
               🎉 no goals
             -/


theorem one_le {a : Associates M} : 1 ≤ a :=
  Dvd.intro _ (one_mul a)


theorem le_mul_right {a b : Associates M} : a ≤ a * b :=
  ⟨b, rfl⟩


                                                           /-
                                                             M : Type u_1
                                                             inst✝ : CommMonoid M
                                                             a b : Associates M
                                                             ⊢ LE.le a (HMul.hMul b a)
                                                           -/
theorem le_mul_left {a b : Associates M} : a ≤ b * a := by rw [mul_comm]; exact le_mul_right
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


instance instOrderBot : OrderBot (Associates M) where
  bot := 1
  bot_le _ := one_le


@[simp]
theorem mk_dvd_mk {a b : M} : Associates.mk a ∣ Associates.mk b ↔ a ∣ b := by
  simp only [dvd_def, mk_surjective.exists, mk_mul_mk, mk_eq_mk_iff_associated,
    Associated.comm (x := b)]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a b : M
    ⊢ Iff (Exists fun x => Associated (HMul.hMul a x) b) (Exists fun c => Eq b (HM …
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoid M
      a b : M
      ⊢ (Exists fun x => Associated (HMul.hMul a x) b) → Exists fun c => Eq b (HMul. …
    -/
  · rintro ⟨x, u, rfl⟩
    /-
      case mp.intro.intro
      M : Type u_1
      inst✝ : CommMonoid M
      a x : M
      u : Units M
      ⊢ Exists fun c => Eq (HMul.hMul (HMul.hMul a x) ↑u) (HMul.hMul a c)
    -/
    exact ⟨_, mul_assoc ..⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      M : Type u_1
      inst✝ : CommMonoid M
      a b : M
      ⊢ (Exists fun c => Eq b (HMul.hMul a c)) → Exists fun x => Associated (HMul.hM …
    -/
  · rintro ⟨c, rfl⟩
    /-
      case mpr.intro
      M : Type u_1
      inst✝ : CommMonoid M
      a c : M
      ⊢ Exists fun x => Associated (HMul.hMul a x) (HMul.hMul a c)
    -/
    use c
    /-
      🎉 no goals
    -/


theorem dvd_of_mk_le_mk {a b : M} : Associates.mk a ≤ Associates.mk b → a ∣ b :=
  mk_dvd_mk.mp


theorem mk_le_mk_of_dvd {a b : M} : a ∣ b → Associates.mk a ≤ Associates.mk b :=
  mk_dvd_mk.mpr


theorem mk_le_mk_iff_dvd {a b : M} : Associates.mk a ≤ Associates.mk b ↔ a ∣ b := mk_dvd_mk


@[deprecated (since := "2024-03-16")] alias mk_le_mk_iff_dvd_iff := mk_le_mk_iff_dvd


@[simp]
theorem isPrimal_mk {a : M} : IsPrimal (Associates.mk a) ↔ IsPrimal a := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : M
    ⊢ Iff (IsPrimal (Associates.mk a)) (IsPrimal a)
  -/
  simp_rw [IsPrimal, forall_associated, mk_surjective.exists, mk_mul_mk, mk_dvd_mk]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a : M
    ⊢ Iff (∀ (a_1 a_2 : M), Dvd.dvd a (HMul.hMul a_1 a_2) → Exists fun x => Exists …
  -/
  constructor <;> intro h b c dvd <;> obtain ⟨a₁, a₂, h₁, h₂, eq⟩ := @h b c dvd
    /-
      case mp.intro.intro.intro.intro
      M : Type u_1
      inst✝ : CommMonoid M
      a : M
      h : ∀ (a_1 a_2 : M), Dvd.dvd a (HMul.hMul a_1 a_2) → Exists fun x => Exists fu …
      b c : M
      dvd : Dvd.dvd a (HMul.hMul b c)
      a₁ a₂ : M
      h₁ : Dvd.dvd a₁ b
      h₂ : Dvd.dvd a₂ c
      eq : Eq (Associates.mk a) (Associates.mk (HMul.hMul a₁ a₂))
      ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
    -/
  · obtain ⟨u, rfl⟩ := mk_eq_mk_iff_associated.mp eq.symm
    /-
      case mp.intro.intro.intro.intro.intro
      M : Type u_1
      inst✝ : CommMonoid M
      b c a₁ a₂ : M
      h₁ : Dvd.dvd a₁ b
      h₂ : Dvd.dvd a₂ c
      u : Units M
      h : ∀ (a a_1 : M), Dvd.dvd (HMul.hMul (HMul.hMul a₁ a₂) ↑u) (HMul.hMul a a_1)  …
      dvd : Dvd.dvd (HMul.hMul (HMul.hMul a₁ a₂) ↑u) (HMul.hMul b c)
      eq : Eq (Associates.mk (HMul.hMul (HMul.hMul a₁ a₂) ↑u)) (Associates.mk (HMul. …
      ⊢ Exists fun a₁_1 => Exists fun a₂_1 => And (Dvd.dvd a₁_1 b) (And (Dvd.dvd a₂_ …
    -/
    exact ⟨a₁, a₂ * u, h₁, Units.mul_right_dvd.mpr h₂, mul_assoc _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.intro.intro.intro
      M : Type u_1
      inst✝ : CommMonoid M
      a : M
      h : ∀ ⦃b c : M⦄, Dvd.dvd a (HMul.hMul b c) → Exists fun a₁ => Exists fun a₂ => …
      b c : M
      dvd : Dvd.dvd a (HMul.hMul b c)
      a₁ a₂ : M
      h₁ : Dvd.dvd a₁ b
      h₂ : Dvd.dvd a₂ c
      eq : Eq a (HMul.hMul a₁ a₂)
      ⊢ Exists fun x => Exists fun x_1 => And (Dvd.dvd x b) (And (Dvd.dvd x_1 c) (Eq …
    -/
  · exact ⟨a₁, a₂, h₁, h₂, congr_arg _ eq⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-03-16")] alias isPrimal_iff := isPrimal_mk


@[simp]
theorem decompositionMonoid_iff : DecompositionMonoid (Associates M) ↔ DecompositionMonoid M := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ⊢ Iff (DecompositionMonoid (Associates M)) (DecompositionMonoid M)
  -/
  simp_rw [_root_.decompositionMonoid_iff, forall_associated, isPrimal_mk]
  /-
    🎉 no goals
  -/


instance instDecompositionMonoid [DecompositionMonoid M] : DecompositionMonoid (Associates M) :=
  decompositionMonoid_iff.mpr ‹_›


@[simp]
theorem mk_isRelPrime_iff {a b : M} :
    IsRelPrime (Associates.mk a) (Associates.mk b) ↔ IsRelPrime a b := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    a b : M
    ⊢ Iff (IsRelPrime (Associates.mk a) (Associates.mk b)) (IsRelPrime a b)
  -/
  simp_rw [IsRelPrime, forall_associated, mk_dvd_mk, isUnit_mk]
  /-
    🎉 no goals
  -/


instance [Zero M] [Monoid M] : Zero (Associates M) :=
  ⟨⟦0⟧⟩


instance [Zero M] [Monoid M] : Top (Associates M) :=
  ⟨0⟩


@[simp] theorem mk_zero [Zero M] [Monoid M] : Associates.mk (0 : M) = 0 := rfl


@[simp]
theorem mk_eq_zero {a : M} : Associates.mk a = 0 ↔ a = 0 :=
  ⟨fun h => (associated_zero_iff_eq_zero a).1 <| Quotient.exact h, fun h => h.symm ▸ rfl⟩


@[simp]
                                                              /-
                                                                M : Type u_1
                                                                inst✝ : MonoidWithZero M
                                                                ⊢ Eq (Quot.out 0) 0
                                                              -/
theorem quot_out_zero : Quot.out (0 : Associates M) = 0 := by rw [← mk_eq_zero, quot_out]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem mk_ne_zero {a : M} : Associates.mk a ≠ 0 ↔ a ≠ 0 :=
  not_congr mk_eq_zero


instance [Nontrivial M] : Nontrivial (Associates M) :=
  ⟨⟨1, 0, mk_ne_zero.2 one_ne_zero⟩⟩


theorem exists_non_zero_rep {a : Associates M} : a ≠ 0 → ∃ a0 : M, a0 ≠ 0 ∧ Associates.mk a0 = a :=
  Quotient.inductionOn a fun b nz => ⟨b, mt (congr_arg Quotient.mk'') nz, rfl⟩


instance instCommMonoidWithZero : CommMonoidWithZero (Associates M) where
                                               /-
                                                 M : Type u_1
                                                 inst✝ : CommMonoidWithZero M
                                                 a : M
                                                 ⊢ Eq (HMul.hMul 0 (Associates.mk a)) 0
                                               -/
    zero_mul := forall_associated.2 fun a ↦ by rw [← mk_zero, mk_mul_mk, zero_mul]
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 M : Type u_1
                                                 inst✝ : CommMonoidWithZero M
                                                 a : M
                                                 ⊢ Eq (HMul.hMul (Associates.mk a) 0) 0
                                               -/
    mul_zero := forall_associated.2 fun a ↦ by rw [← mk_zero, mk_mul_mk, mul_zero]
                                               /-
                                                 🎉 no goals
                                               -/


instance instOrderTop : OrderTop (Associates M) where
  top := 0
  le_top := dvd_zero


@[simp] protected theorem le_zero (a : Associates M) : a ≤ 0 := le_top


instance instBoundedOrder : BoundedOrder (Associates M) where


instance [DecidableRel ((· ∣ ·) : M → M → Prop)] :
    DecidableRel ((· ∣ ·) : Associates M → Associates M → Prop) := fun a b =>
  Quotient.recOnSubsingleton₂ a b fun _ _ => decidable_of_iff' _ mk_dvd_mk


theorem Prime.le_or_le {p : Associates M} (hp : Prime p) {a b : Associates M} (h : p ≤ a * b) :
    p ≤ a ∨ p ≤ b :=
  hp.2.2 a b h


@[simp]
theorem prime_mk {p : M} : Prime (Associates.mk p) ↔ Prime p := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p : M
    ⊢ Iff (Prime (Associates.mk p)) (Prime p)
  -/
  rw [Prime, _root_.Prime, forall_associated]
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    p : M
    ⊢ Iff (And (Ne (Associates.mk p) 0) (And (Not (IsUnit (Associates.mk p))) (∀ ( …
  -/
  simp only [forall_associated, mk_ne_zero, isUnit_mk, mk_mul_mk, mk_dvd_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem irreducible_mk {a : M} : Irreducible (Associates.mk a) ↔ Irreducible a := by
  simp only [irreducible_iff, isUnit_mk, forall_associated, isUnit_mk, mk_mul_mk,
    mk_eq_mk_iff_associated, Associated.comm (x := a)]
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a : M
    ⊢ Iff (And (Not (IsUnit a)) (∀ (a_1 a_2 : M), Associated (HMul.hMul a_1 a_2) a …
  -/
  apply Iff.rfl.and
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a : M
    ⊢ Iff (∀ (a_1 a_2 : M), Associated (HMul.hMul a_1 a_2) a → Or (IsUnit a_1) (Is …
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a : M
      ⊢ (∀ (a_1 a_2 : M), Associated (HMul.hMul a_1 a_2) a → Or (IsUnit a_1) (IsUnit …
    -/
  · rintro h x y rfl
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      x y : M
      h : ∀ (a a_1 : M), Associated (HMul.hMul a a_1) (HMul.hMul x y) → Or (IsUnit a …
      ⊢ Or (IsUnit x) (IsUnit y)
    -/
    exact h _ _ <| .refl _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a : M
      ⊢ (∀ (a_1 b : M), Eq a (HMul.hMul a_1 b) → Or (IsUnit a_1) (IsUnit b)) → ∀ (a_ …
    -/
  · rintro h x y ⟨u, rfl⟩
    /-
      case mpr.intro
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      x y : M
      u : Units M
      h : ∀ (a b : M), Eq (HMul.hMul (HMul.hMul x y) ↑u) (HMul.hMul a b) → Or (IsUni …
      ⊢ Or (IsUnit x) (IsUnit y)
    -/
    simpa using h x (y * u) (mul_assoc _ _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem mk_dvdNotUnit_mk_iff {a b : M} :
    DvdNotUnit (Associates.mk a) (Associates.mk b) ↔ DvdNotUnit a b := by
  simp only [DvdNotUnit, mk_ne_zero, mk_surjective.exists, isUnit_mk, mk_mul_mk,
    mk_eq_mk_iff_associated, Associated.comm (x := b)]
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a b : M
    ⊢ Iff (And (Ne a 0) (Exists fun x => And (Not (IsUnit x)) (Associated (HMul.hM …
  -/
  refine Iff.rfl.and ?_
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a b : M
    ⊢ Iff (Exists fun x => And (Not (IsUnit x)) (Associated (HMul.hMul a x) b)) (E …
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a b : M
      ⊢ (Exists fun x => And (Not (IsUnit x)) (Associated (HMul.hMul a x) b)) → Exis …
    -/
  · rintro ⟨x, hx, u, rfl⟩
    /-
      case mp.intro.intro.intro
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a x : M
      hx : Not (IsUnit x)
      u : Units M
      ⊢ Exists fun x_1 => And (Not (IsUnit x_1)) (Eq (HMul.hMul (HMul.hMul a x) ↑u)  …
    -/
    refine ⟨x * u, ?_, mul_assoc ..⟩
    /-
      case mp.intro.intro.intro
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a x : M
      hx : Not (IsUnit x)
      u : Units M
      ⊢ Not (IsUnit (HMul.hMul x ↑u))
    -/
    simpa
    /-
      🎉 no goals
    -/
    /-
      case mpr
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a b : M
      ⊢ (Exists fun x => And (Not (IsUnit x)) (Eq b (HMul.hMul a x))) → Exists fun x …
    -/
  · rintro ⟨x, ⟨hx, rfl⟩⟩
    /-
      case mpr.intro.intro
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a x : M
      hx : Not (IsUnit x)
      ⊢ Exists fun x_1 => And (Not (IsUnit x_1)) (Associated (HMul.hMul a x_1) (HMul …
    -/
    use x
    /-
      🎉 no goals
    -/


theorem dvdNotUnit_of_lt {a b : Associates M} (hlt : a < b) : DvdNotUnit a b := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a b : Associates M
    hlt : LT.lt a b
    ⊢ DvdNotUnit a b
  -/
  constructor
    /-
      case left
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      a b : Associates M
      hlt : LT.lt a b
      ⊢ Ne a 0
    -/
  · rintro rfl
    /-
      case left
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      b : Associates M
      hlt : LT.lt 0 b
      ⊢ False
    -/
    apply not_lt_of_le _ hlt
    /-
      M : Type u_1
      inst✝ : CommMonoidWithZero M
      b : Associates M
      hlt : LT.lt 0 b
      ⊢ LE.le b 0
    -/
    apply dvd_zero
    /-
      🎉 no goals
    -/
  /-
    case right
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a b : Associates M
    hlt : LT.lt a b
    ⊢ Exists fun x => And (Not (IsUnit x)) (Eq b (HMul.hMul a x))
  -/
  rcases hlt with ⟨⟨x, rfl⟩, ndvd⟩
  /-
    case right.intro.intro
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a : Associates M
    x : Associates M
    ndvd : Not (Dvd.dvd (HMul.hMul a x) a)
    ⊢ Exists fun x_1 => And (Not (IsUnit x_1)) (Eq (HMul.hMul a x) (HMul.hMul a x_ …
  -/
  refine ⟨x, ?_, rfl⟩
  /-
    case right.intro.intro
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a : Associates M
    x : Associates M
    ndvd : Not (Dvd.dvd (HMul.hMul a x) a)
    ⊢ Not (IsUnit x)
  -/
  contrapose! ndvd
  /-
    case right.intro.intro
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a : Associates M
    x : Associates M
    ndvd : IsUnit x
    ⊢ Dvd.dvd (HMul.hMul a x) a
  -/
  rcases ndvd with ⟨u, rfl⟩
  /-
    case right.intro.intro.intro
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    a : Associates M
    u : Units (Associates M)
    ⊢ Dvd.dvd (HMul.hMul a ↑u) a
  -/
  simp
  /-
    🎉 no goals
  -/


theorem irreducible_iff_prime_iff :
    (∀ a : M, Irreducible a ↔ Prime a) ↔ ∀ a : Associates M, Irreducible a ↔ Prime a := by
  /-
    M : Type u_1
    inst✝ : CommMonoidWithZero M
    ⊢ Iff (∀ (a : M), Iff (Irreducible a) (Prime a)) (∀ (a : Associates M), Iff (I …
  -/
  simp_rw [forall_associated, irreducible_mk, prime_mk]
  /-
    🎉 no goals
  -/


instance instPartialOrder : PartialOrder (Associates M) where
  le_antisymm := mk_surjective.forall₂.2 fun _a _b hab hba => mk_eq_mk_iff_associated.2 <|
    associated_of_dvd_dvd (dvd_of_mk_le_mk hab) (dvd_of_mk_le_mk hba)


instance instCancelCommMonoidWithZero : CancelCommMonoidWithZero (Associates M) :=
        /-
          M : Type u_1
          inst✝ : CancelCommMonoidWithZero M
          ⊢ CommMonoidWithZero (Associates M)
        -/
  { (by infer_instance : CommMonoidWithZero (Associates M)) with
        /-
          🎉 no goals
        -/
    mul_left_cancel_of_ne_zero := by
      /-
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        ⊢ ∀ {a b c : Associates M}, Ne a 0 → Eq (HMul.hMul a b) (HMul.hMul a c) → Eq b c
      -/
      rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ ha h
      /-
        case mk.mk.mk
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        a✝ : Associates M
        a : M
        b✝ : Associates M
        b : M
        c✝ : Associates M
        c : M
        ha : Ne (Quot.mk (⇑(Associated.setoid M)) a) 0
        h : Eq (HMul.hMul (Quot.mk (⇑(Associated.setoid M)) a) (Quot.mk (⇑(Associated. …
        ⊢ Eq (Quot.mk (⇑(Associated.setoid M)) b) (Quot.mk (⇑(Associated.setoid M)) c)
      -/
      rcases Quotient.exact' h with ⟨u, hu⟩
      /-
        case mk.mk.mk.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        a✝ : Associates M
        a : M
        b✝ : Associates M
        b : M
        c✝ : Associates M
        c : M
        ha : Ne (Quot.mk (⇑(Associated.setoid M)) a) 0
        h : Eq (HMul.hMul (Quot.mk (⇑(Associated.setoid M)) a) (Quot.mk (⇑(Associated. …
        u : Units M
        hu : Eq (HMul.hMul ((fun x1 x2 => HMul.hMul x1 x2) a b) ↑u) ((fun x1 x2 => HMu …
        ⊢ Eq (Quot.mk (⇑(Associated.setoid M)) b) (Quot.mk (⇑(Associated.setoid M)) c)
      -/
      have hu : a * (b * ↑u) = a * c := by rwa [← mul_assoc]
      /-
        case mk.mk.mk.intro
        M : Type u_1
        inst✝ : CancelCommMonoidWithZero M
        a✝ : Associates M
        a : M
        b✝ : Associates M
        b : M
        c✝ : Associates M
        c : M
        ha : Ne (Quot.mk (⇑(Associated.setoid M)) a) 0
        h : Eq (HMul.hMul (Quot.mk (⇑(Associated.setoid M)) a) (Quot.mk (⇑(Associated. …
        u : Units M
        hu✝ : Eq (HMul.hMul ((fun x1 x2 => HMul.hMul x1 x2) a b) ↑u) ((fun x1 x2 => HM …
        hu : Eq (HMul.hMul a (HMul.hMul b ↑u)) (HMul.hMul a c)
        ⊢ Eq (Quot.mk (⇑(Associated.setoid M)) b) (Quot.mk (⇑(Associated.setoid M)) c)
      -/
      exact Quotient.sound' ⟨u, mul_left_cancel₀ (mk_ne_zero.1 ha) hu⟩ }
      /-
        🎉 no goals
      -/


theorem _root_.associates_irreducible_iff_prime [DecompositionMonoid M] {p : Associates M} :
    Irreducible p ↔ Prime p := irreducible_iff_prime


                                               /-
                                                 M : Type u_1
                                                 inst✝ : CancelCommMonoidWithZero M
                                                 ⊢ NoZeroDivisors (Associates M)
                                               -/
instance : NoZeroDivisors (Associates M) := by infer_instance
                                               /-
                                                 🎉 no goals
                                               -/


theorem le_of_mul_le_mul_left (a b c : Associates M) (ha : a ≠ 0) : a * b ≤ a * c → b ≤ c
                                             /-
                                               M : Type u_1
                                               inst✝ : CancelCommMonoidWithZero M
                                               a b c : Associates M
                                               ha : Ne a 0
                                               d : Associates M
                                               hd : Eq (HMul.hMul a c) (HMul.hMul (HMul.hMul a b) d)
                                               ⊢ Eq (HMul.hMul a c) (HMul.hMul a (HMul.hMul b d))
                                             -/
  | ⟨d, hd⟩ => ⟨d, mul_left_cancel₀ ha <| by rwa [← mul_assoc]⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem one_or_eq_of_le_of_prime {p m : Associates M} (hp : Prime p) (hle : m ≤ p) :
    m = 1 ∨ m = p := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p m : Associates M
    hp : Prime p
    hle : LE.le m p
    ⊢ Or (Eq m 1) (Eq m p)
  -/
  rcases mk_surjective p with ⟨p, rfl⟩
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    m : Associates M
    p : M
    hp : Prime (Associates.mk p)
    hle : LE.le m (Associates.mk p)
    ⊢ Or (Eq m 1) (Eq m (Associates.mk p))
  -/
  rcases mk_surjective m with ⟨m, rfl⟩
  simpa [mk_eq_mk_iff_associated, Associated.comm, -Quotient.eq]
    using (prime_mk.1 hp).irreducible.dvd_iff.mp (mk_le_mk_iff_dvd.1 hle)


theorem dvdNotUnit_iff_lt {a b : Associates M} : DvdNotUnit a b ↔ a < b :=
  dvd_and_not_dvd_iff.symm


                                                            /-
                                                              M : Type u_1
                                                              inst✝ : CancelCommMonoidWithZero M
                                                              p : Associates M
                                                              ⊢ Iff (LE.le p 1) (Eq p 1)
                                                            -/
theorem le_one_iff {p : Associates M} : p ≤ 1 ↔ p = 1 := by rw [← Associates.bot_eq_one, le_bot_iff]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem dvdNotUnit_of_dvdNotUnit_associated [CommMonoidWithZero M] [Nontrivial M] {p q r : M}
    (h : DvdNotUnit p q) (h' : Associated q r) : DvdNotUnit p r := by
  /-
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    inst✝ : Nontrivial M
    p q r : M
    h : DvdNotUnit p q
    h' : Associated q r
    ⊢ DvdNotUnit p r
  -/
  obtain ⟨u, rfl⟩ := Associated.symm h'
  /-
    case intro
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    inst✝ : Nontrivial M
    p r : M
    u : Units M
    h : DvdNotUnit p (HMul.hMul r ↑u)
    h' : Associated (HMul.hMul r ↑u) r
    ⊢ DvdNotUnit p r
  -/
  obtain ⟨hp, x, hx⟩ := h
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    inst✝ : Nontrivial M
    p r : M
    u : Units M
    h' : Associated (HMul.hMul r ↑u) r
    hp : Ne p 0
    x : M
    hx : And (Not (IsUnit x)) (Eq (HMul.hMul r ↑u) (HMul.hMul p x))
    ⊢ DvdNotUnit p r
  -/
  refine ⟨hp, x * ↑u⁻¹, DvdNotUnit.not_unit ⟨u⁻¹.ne_zero, x, hx.left, mul_comm _ _⟩, ?_⟩
  /-
    case intro.intro.intro
    M : Type u_1
    inst✝¹ : CommMonoidWithZero M
    inst✝ : Nontrivial M
    p r : M
    u : Units M
    h' : Associated (HMul.hMul r ↑u) r
    hp : Ne p 0
    x : M
    hx : And (Not (IsUnit x)) (Eq (HMul.hMul r ↑u) (HMul.hMul p x))
    ⊢ Eq r (HMul.hMul p (HMul.hMul x ↑(Inv.inv u)))
  -/
  rw [← mul_assoc, ← hx.right, mul_assoc, Units.mul_inv, mul_one]
  /-
    🎉 no goals
  -/


theorem isUnit_of_associated_mul [CancelCommMonoidWithZero M] {p b : M} (h : Associated (p * b) p)
    (hp : p ≠ 0) : IsUnit b := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p b : M
    h : Associated (HMul.hMul p b) p
    hp : Ne p 0
    ⊢ IsUnit b
  -/
  obtain ⟨a, ha⟩ := h
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p b : M
    hp : Ne p 0
    a : Units M
    ha : Eq (HMul.hMul (HMul.hMul p b) ↑a) p
    ⊢ IsUnit b
  -/
  refine isUnit_of_mul_eq_one b a ((mul_right_inj' hp).mp ?_)
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p b : M
    hp : Ne p 0
    a : Units M
    ha : Eq (HMul.hMul (HMul.hMul p b) ↑a) p
    ⊢ Eq (HMul.hMul p (HMul.hMul b ↑a)) (HMul.hMul p 1)
  -/
  rwa [← mul_assoc, mul_one]
  /-
    🎉 no goals
  -/


theorem DvdNotUnit.not_associated [CancelCommMonoidWithZero M] {p q : M} (h : DvdNotUnit p q) :
    ¬Associated p q := by
  /-
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p q : M
    h : DvdNotUnit p q
    ⊢ Not (Associated p q)
  -/
  rintro ⟨a, rfl⟩
  /-
    case intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    a : Units M
    h : DvdNotUnit p (HMul.hMul p ↑a)
    ⊢ False
  -/
  obtain ⟨hp, x, hx, hx'⟩ := h
  /-
    case intro.intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    a : Units M
    hp : Ne p 0
    x : M
    hx : Not (IsUnit x)
    hx' : Eq (HMul.hMul p ↑a) (HMul.hMul p x)
    ⊢ False
  -/
  rcases (mul_right_inj' hp).mp hx' with rfl
  /-
    case intro.intro.intro.intro
    M : Type u_1
    inst✝ : CancelCommMonoidWithZero M
    p : M
    a : Units M
    hp : Ne p 0
    hx : Not (IsUnit ↑a)
    hx' : Eq (HMul.hMul p ↑a) (HMul.hMul p ↑a)
    ⊢ False
  -/
  exact hx a.isUnit
  /-
    🎉 no goals
  -/


theorem dvd_prime_pow [CancelCommMonoidWithZero M] {p q : M} (hp : Prime p) (n : ℕ) :
    q ∣ p ^ n ↔ ∃ i ≤ n, Associated q (p ^ i) := by
  induction n generalizing q with
  | zero =>
    simp [← isUnit_iff_dvd_one, associated_one_iff_isUnit]
  | succ n ih =>
    refine ⟨fun h => ?_, fun ⟨i, hi, hq⟩ => hq.dvd.trans (pow_dvd_pow p hi)⟩
    rw [pow_succ'] at h
    rcases hp.left_dvd_or_dvd_right_of_dvd_mul h with (⟨q, rfl⟩ | hno)
    · rw [mul_dvd_mul_iff_left hp.ne_zero, ih] at h
      rcases h with ⟨i, hi, hq⟩
      refine ⟨i + 1, Nat.succ_le_succ hi, (hq.mul_left p).trans ?_⟩
      rw [pow_succ']
    · obtain ⟨i, hi, hq⟩ := ih.mp hno
      exact ⟨i, hi.trans n.le_succ, hq⟩


