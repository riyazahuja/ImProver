/-- An element is said to be nilpotent if some natural-number-power of it equals zero.

Note that we require only the bare minimum assumptions for the definition to make sense. Even
`MonoidWithZero` is too strong since nilpotency is important in the study of rings that are only
power-associative. -/
def IsNilpotent [Zero R] [Pow R ℕ] (x : R) : Prop :=
  ∃ n : ℕ, x ^ n = 0


theorem IsNilpotent.mk [Zero R] [Pow R ℕ] (x : R) (n : ℕ) (e : x ^ n = 0) : IsNilpotent x :=
  ⟨n, e⟩


@[simp] lemma isNilpotent_of_subsingleton [Zero R] [Pow R ℕ] [Subsingleton R] : IsNilpotent x :=
  ⟨0, Subsingleton.elim _ _⟩


@[simp] theorem IsNilpotent.zero [MonoidWithZero R] : IsNilpotent (0 : R) :=
  ⟨1, pow_one 0⟩


theorem not_isNilpotent_one [MonoidWithZero R] [Nontrivial R] :
    ¬ IsNilpotent (1 : R) := fun ⟨_, H⟩ ↦ zero_ne_one (H.symm.trans (one_pow _))


lemma IsNilpotent.pow_succ (n : ℕ) {S : Type*} [MonoidWithZero S] {x : S}
    (hx : IsNilpotent x) : IsNilpotent (x ^ n.succ) := by
  /-
    n : Nat
    S : Type u_3
    inst✝ : MonoidWithZero S
    x : S
    hx : IsNilpotent x
    ⊢ IsNilpotent (HPow.hPow x n.succ)
  -/
  obtain ⟨N,hN⟩ := hx
  /-
    case intro
    n : Nat
    S : Type u_3
    inst✝ : MonoidWithZero S
    x : S
    N : Nat
    hN : Eq (HPow.hPow x N) 0
    ⊢ IsNilpotent (HPow.hPow x n.succ)
  -/
  use N
  /-
    case h
    n : Nat
    S : Type u_3
    inst✝ : MonoidWithZero S
    x : S
    N : Nat
    hN : Eq (HPow.hPow x N) 0
    ⊢ Eq (HPow.hPow (HPow.hPow x n.succ) N) 0
  -/
  rw [← pow_mul, Nat.succ_mul, pow_add, hN, mul_zero]
  /-
    🎉 no goals
  -/


theorem  IsNilpotent.of_pow [MonoidWithZero R] {x : R} {m : ℕ}
    (h : IsNilpotent (x ^ m)) : IsNilpotent x := by
  /-
    R : Type u_1
    inst✝ : MonoidWithZero R
    x : R
    m : Nat
    h : IsNilpotent (HPow.hPow x m)
    ⊢ IsNilpotent x
  -/
  obtain ⟨n, h⟩ := h
  /-
    case intro
    R : Type u_1
    inst✝ : MonoidWithZero R
    x : R
    m n : Nat
    h : Eq (HPow.hPow (HPow.hPow x m) n) 0
    ⊢ IsNilpotent x
  -/
  use m*n
  /-
    case h
    R : Type u_1
    inst✝ : MonoidWithZero R
    x : R
    m n : Nat
    h : Eq (HPow.hPow (HPow.hPow x m) n) 0
    ⊢ Eq (HPow.hPow x (HMul.hMul m n)) 0
  -/
  rw [← h, pow_mul x m n]
  /-
    🎉 no goals
  -/


lemma IsNilpotent.pow_of_pos {n} {S : Type*} [MonoidWithZero S] {x : S}
    (hx : IsNilpotent x) (hn : n ≠ 0) : IsNilpotent (x ^ n) := by
  cases n with
  | zero => contradiction
  | succ => exact  IsNilpotent.pow_succ _ hx


@[simp]
lemma IsNilpotent.pow_iff_pos {n} {S : Type*} [MonoidWithZero S] {x : S}
    (hn : n ≠ 0) : IsNilpotent (x ^ n) ↔ IsNilpotent x :=
 ⟨fun h => of_pow h, fun h => pow_of_pos h hn⟩


theorem IsNilpotent.map [MonoidWithZero R] [MonoidWithZero S] {r : R} {F : Type*}
    [FunLike F R S] [MonoidWithZeroHomClass F R S] (hr : IsNilpotent r) (f : F) :
    IsNilpotent (f r) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : MonoidWithZero R
    inst✝² : MonoidWithZero S
    r : R
    F : Type u_3
    inst✝¹ : FunLike F R S
    inst✝ : MonoidWithZeroHomClass F R S
    hr : IsNilpotent r
    f : F
    ⊢ IsNilpotent (f r)
  -/
  use hr.choose
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝³ : MonoidWithZero R
    inst✝² : MonoidWithZero S
    r : R
    F : Type u_3
    inst✝¹ : FunLike F R S
    inst✝ : MonoidWithZeroHomClass F R S
    hr : IsNilpotent r
    f : F
    ⊢ Eq (HPow.hPow (f r) (Exists.choose hr)) 0
  -/
  rw [← map_pow, hr.choose_spec, map_zero]
  /-
    🎉 no goals
  -/


lemma IsNilpotent.map_iff [MonoidWithZero R] [MonoidWithZero S] {r : R} {F : Type*}
    [FunLike F R S] [MonoidWithZeroHomClass F R S] {f : F} (hf : Function.Injective f) :
    IsNilpotent (f r) ↔ IsNilpotent r :=
                                                     /-
                                                       R : Type u_1
                                                       S : Type u_2
                                                       inst✝³ : MonoidWithZero R
                                                       inst✝² : MonoidWithZero S
                                                       r : R
                                                       F : Type u_3
                                                       inst✝¹ : FunLike F R S
                                                       inst✝ : MonoidWithZeroHomClass F R S
                                                       f : F
                                                       hf : Function.Injective ⇑f
                                                       x✝ : IsNilpotent (f r)
                                                       k : Nat
                                                       hk : Eq (HPow.hPow (f r) k) 0
                                                       ⊢ Eq (f (HPow.hPow r k)) 0
                                                     -/
  ⟨fun ⟨k, hk⟩ ↦ ⟨k, (map_eq_zero_iff f hf).mp <| by rwa [map_pow]⟩, fun h ↦ h.map f⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem IsUnit.isNilpotent_mul_unit_of_commute_iff [MonoidWithZero R] {r u : R}
    (hu : IsUnit u) (h_comm : Commute r u) :
    IsNilpotent (r * u) ↔ IsNilpotent r :=
                          /-
                            R : Type u_1
                            inst✝ : MonoidWithZero R
                            r u : R
                            hu : IsUnit u
                            h_comm : Commute r u
                            n : Nat
                            ⊢ Iff (Eq (HPow.hPow (HMul.hMul r u) n) 0) (Eq (HPow.hPow r n) 0)
                          -/
  exists_congr fun n ↦ by rw [h_comm.mul_pow, (hu.pow n).mul_left_eq_zero]
                          /-
                            🎉 no goals
                          -/


theorem IsUnit.isNilpotent_unit_mul_of_commute_iff [MonoidWithZero R] {r u : R}
    (hu : IsUnit u) (h_comm : Commute r u) :
    IsNilpotent (u * r) ↔ IsNilpotent r :=
  h_comm ▸ hu.isNilpotent_mul_unit_of_commute_iff h_comm


variable (x) in
/-- If `x` is nilpotent, the nilpotency class is the smallest natural number `k` such that
`x ^ k = 0`. If `x` is not nilpotent, the nilpotency class takes the junk value `0`. -/
noncomputable def nilpotencyClass : ℕ := sInf {k | x ^ k = 0}


@[simp] lemma nilpotencyClass_eq_zero_of_subsingleton [Subsingleton R] :
    nilpotencyClass x = 0 := by
  /-
    R : Type u_1
    x : R
    inst✝² : Zero R
    inst✝¹ : Pow R Nat
    inst✝ : Subsingleton R
    ⊢ Eq (nilpotencyClass x) 0
  -/
  let s : Set ℕ := {k | x ^ k = 0}
  /-
    R : Type u_1
    x : R
    inst✝² : Zero R
    inst✝¹ : Pow R Nat
    inst✝ : Subsingleton R
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    ⊢ Eq (nilpotencyClass x) 0
  -/
  suffices s = univ by change sInf _ = 0; simp [s] at this; simp [this]
  /-
    R : Type u_1
    x : R
    inst✝² : Zero R
    inst✝¹ : Pow R Nat
    inst✝ : Subsingleton R
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    ⊢ Eq s Set.univ
  -/
  exact eq_univ_iff_forall.mpr fun k ↦ Subsingleton.elim _ _
  /-
    🎉 no goals
  -/


lemma isNilpotent_of_pos_nilpotencyClass (hx : 0 < nilpotencyClass x) :
    IsNilpotent x := by
  /-
    R : Type u_1
    x : R
    inst✝¹ : Zero R
    inst✝ : Pow R Nat
    hx : LT.lt 0 (nilpotencyClass x)
    ⊢ IsNilpotent x
  -/
  let s : Set ℕ := {k | x ^ k = 0}
  /-
    R : Type u_1
    x : R
    inst✝¹ : Zero R
    inst✝ : Pow R Nat
    hx : LT.lt 0 (nilpotencyClass x)
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    ⊢ IsNilpotent x
  -/
  change s.Nonempty
  /-
    R : Type u_1
    x : R
    inst✝¹ : Zero R
    inst✝ : Pow R Nat
    hx : LT.lt 0 (nilpotencyClass x)
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    ⊢ s.Nonempty
  -/
  change 0 < sInf s at hx
  /-
    R : Type u_1
    x : R
    inst✝¹ : Zero R
    inst✝ : Pow R Nat
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    hx : LT.lt 0 (InfSet.sInf s)
    ⊢ s.Nonempty
  -/
  by_contra contra
  /-
    R : Type u_1
    x : R
    inst✝¹ : Zero R
    inst✝ : Pow R Nat
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    hx : LT.lt 0 (InfSet.sInf s)
    contra : Not s.Nonempty
    ⊢ False
  -/
  simp [not_nonempty_iff_eq_empty.mp contra] at hx
  /-
    🎉 no goals
  -/


lemma pow_nilpotencyClass (hx : IsNilpotent x) : x ^ (nilpotencyClass x) = 0 :=
  Nat.sInf_mem hx


lemma nilpotencyClass_eq_succ_iff {k : ℕ} :
    nilpotencyClass x = k + 1 ↔ x ^ (k + 1) = 0 ∧ x ^ k ≠ 0 := by
  /-
    R : Type u_1
    x : R
    inst✝ : MonoidWithZero R
    k : Nat
    ⊢ Iff (Eq (nilpotencyClass x) (HAdd.hAdd k 1)) (And (Eq (HPow.hPow x (HAdd.hAd …
  -/
  let s : Set ℕ := {k | x ^ k = 0}
  /-
    R : Type u_1
    x : R
    inst✝ : MonoidWithZero R
    k : Nat
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    ⊢ Iff (Eq (nilpotencyClass x) (HAdd.hAdd k 1)) (And (Eq (HPow.hPow x (HAdd.hAd …
  -/
  have : ∀ k₁ k₂ : ℕ, k₁ ≤ k₂ → k₁ ∈ s → k₂ ∈ s := fun k₁ k₂ h_le hk₁ ↦ pow_eq_zero_of_le h_le hk₁
  /-
    R : Type u_1
    x : R
    inst✝ : MonoidWithZero R
    k : Nat
    s : Set Nat := setOf fun k => Eq (HPow.hPow x k) 0
    this : ∀ (k₁ k₂ : Nat), LE.le k₁ k₂ → Membership.mem s k₁ → Membership.mem s k₂
    ⊢ Iff (Eq (nilpotencyClass x) (HAdd.hAdd k 1)) (And (Eq (HPow.hPow x (HAdd.hAd …
  -/
  simp [s, nilpotencyClass, Nat.sInf_upward_closed_eq_succ_iff this]
  /-
    🎉 no goals
  -/


@[simp] lemma nilpotencyClass_zero [Nontrivial R] :
    nilpotencyClass (0 : R) = 1 :=
                                        /-
                                          R : Type u_1
                                          inst✝¹ : MonoidWithZero R
                                          inst✝ : Nontrivial R
                                          ⊢ And (Eq (HPow.hPow 0 (HAdd.hAdd 0 1)) 0) (Ne (HPow.hPow 0 0) 0)
                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  nilpotencyClass_eq_succ_iff.mpr <| by constructor <;> simp
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp] lemma pos_nilpotencyClass_iff [Nontrivial R] :
    0 < nilpotencyClass x ↔ IsNilpotent x := by
  /-
    R : Type u_1
    x : R
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    ⊢ Iff (LT.lt 0 (nilpotencyClass x)) (IsNilpotent x)
  -/
  refine ⟨isNilpotent_of_pos_nilpotencyClass, fun hx ↦ Nat.pos_of_ne_zero fun hx' ↦ ?_⟩
  /-
    R : Type u_1
    x : R
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    hx : IsNilpotent x
    hx' : Eq (nilpotencyClass x) 0
    ⊢ False
  -/
  replace hx := pow_nilpotencyClass hx
  /-
    R : Type u_1
    x : R
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    hx' : Eq (nilpotencyClass x) 0
    hx : Eq (HPow.hPow x (nilpotencyClass x)) 0
    ⊢ False
  -/
  rw [hx', pow_zero] at hx
  /-
    R : Type u_1
    x : R
    inst✝¹ : MonoidWithZero R
    inst✝ : Nontrivial R
    hx' : Eq (nilpotencyClass x) 0
    hx : Eq 1 0
    ⊢ False
  -/
  exact one_ne_zero hx
  /-
    🎉 no goals
  -/


lemma pow_pred_nilpotencyClass [Nontrivial R] (hx : IsNilpotent x) :
    x ^ (nilpotencyClass x - 1) ≠ 0 :=
  (nilpotencyClass_eq_succ_iff.mp <| Nat.eq_add_of_sub_eq (pos_nilpotencyClass_iff.mpr hx) rfl).2


lemma eq_zero_of_nilpotencyClass_eq_one (hx : nilpotencyClass x = 1) :
    x = 0 := by
  /-
    R : Type u_1
    x : R
    inst✝ : MonoidWithZero R
    hx : Eq (nilpotencyClass x) 1
    ⊢ Eq x 0
  -/
  have : IsNilpotent x := isNilpotent_of_pos_nilpotencyClass (hx ▸ Nat.one_pos)
  /-
    R : Type u_1
    x : R
    inst✝ : MonoidWithZero R
    hx : Eq (nilpotencyClass x) 1
    this : IsNilpotent x
    ⊢ Eq x 0
  -/
  rw [← pow_nilpotencyClass this, hx, pow_one]
  /-
    🎉 no goals
  -/


@[simp] lemma nilpotencyClass_eq_one [Nontrivial R] :
    nilpotencyClass x = 1 ↔ x = 0 :=
  ⟨eq_zero_of_nilpotencyClass_eq_one, fun hx ↦ hx ▸ nilpotencyClass_zero⟩


/-- A structure that has zero and pow is reduced if it has no nonzero nilpotent elements. -/
@[mk_iff]
class IsReduced (R : Type*) [Zero R] [Pow R ℕ] : Prop where
  /-- A reduced structure has no nonzero nilpotent elements. -/
  eq_zero : ∀ x : R, IsNilpotent x → x = 0


theorem pow_eq_zero [Zero R] [Pow R ℕ] [IsReduced R] {n : ℕ} (h : x ^ n = 0) :
    x = 0 := IsReduced.eq_zero x ⟨n, h⟩


@[simp]
theorem pow_eq_zero_iff [MonoidWithZero R] [IsReduced R] {n : ℕ} (hn : n ≠ 0) :
    x ^ n = 0 ↔ x = 0 := ⟨pow_eq_zero, fun h ↦ h.symm ▸ zero_pow hn⟩


theorem pow_ne_zero_iff [MonoidWithZero R] [IsReduced R] {n : ℕ} (hn : n ≠ 0) :
    x ^ n ≠ 0 ↔ x ≠ 0 := not_congr (pow_eq_zero_iff hn)


theorem pow_ne_zero [Zero R] [Pow R ℕ] [IsReduced R] (n : ℕ) (h : x ≠ 0) :
    x ^ n ≠ 0 := fun H ↦ h (pow_eq_zero H)


/-- A variant of `IsReduced.pow_eq_zero_iff` assuming `R` is not trivial. -/
@[simp]
theorem pow_eq_zero_iff' [MonoidWithZero R] [IsReduced R] [Nontrivial R] {n : ℕ} :
    x ^ n = 0 ↔ x = 0 ∧ n ≠ 0 := by
  /-
    R : Type u_1
    x : R
    inst✝² : MonoidWithZero R
    inst✝¹ : IsReduced R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Iff (Eq (HPow.hPow x n) 0) (And (Eq x 0) (Ne n 0))
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


instance (priority := 900) isReduced_of_noZeroDivisors [MonoidWithZero R] [NoZeroDivisors R] :
    IsReduced R :=
  ⟨fun _ ⟨_, hn⟩ => pow_eq_zero hn⟩


instance (priority := 900) isReduced_of_subsingleton [Zero R] [Pow R ℕ] [Subsingleton R] :
    IsReduced R :=
  ⟨fun _ _ => Subsingleton.elim _ _⟩


theorem IsNilpotent.eq_zero [Zero R] [Pow R ℕ] [IsReduced R] (h : IsNilpotent x) : x = 0 :=
  IsReduced.eq_zero x h


@[simp]
theorem isNilpotent_iff_eq_zero [MonoidWithZero R] [IsReduced R] : IsNilpotent x ↔ x = 0 :=
  ⟨fun h => h.eq_zero, fun h => h.symm ▸ IsNilpotent.zero⟩


theorem isReduced_of_injective [MonoidWithZero R] [MonoidWithZero S] {F : Type*}
    [FunLike F R S] [MonoidWithZeroHomClass F R S]
    (f : F) (hf : Function.Injective f) [IsReduced S] :
    IsReduced R := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    F : Type u_3
    inst✝² : FunLike F R S
    inst✝¹ : MonoidWithZeroHomClass F R S
    f : F
    hf : Function.Injective ⇑f
    inst✝ : IsReduced S
    ⊢ IsReduced R
  -/
  constructor
  /-
    case eq_zero
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    F : Type u_3
    inst✝² : FunLike F R S
    inst✝¹ : MonoidWithZeroHomClass F R S
    f : F
    hf : Function.Injective ⇑f
    inst✝ : IsReduced S
    ⊢ ∀ (x : R), IsNilpotent x → Eq x 0
  -/
  intro x hx
  /-
    case eq_zero
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    F : Type u_3
    inst✝² : FunLike F R S
    inst✝¹ : MonoidWithZeroHomClass F R S
    f : F
    hf : Function.Injective ⇑f
    inst✝ : IsReduced S
    x : R
    hx : IsNilpotent x
    ⊢ Eq x 0
  -/
  apply hf
  /-
    case eq_zero.a
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    F : Type u_3
    inst✝² : FunLike F R S
    inst✝¹ : MonoidWithZeroHomClass F R S
    f : F
    hf : Function.Injective ⇑f
    inst✝ : IsReduced S
    x : R
    hx : IsNilpotent x
    ⊢ Eq (f x) (f 0)
  -/
  rw [map_zero]
  /-
    case eq_zero.a
    R : Type u_1
    S : Type u_2
    inst✝⁴ : MonoidWithZero R
    inst✝³ : MonoidWithZero S
    F : Type u_3
    inst✝² : FunLike F R S
    inst✝¹ : MonoidWithZeroHomClass F R S
    f : F
    hf : Function.Injective ⇑f
    inst✝ : IsReduced S
    x : R
    hx : IsNilpotent x
    ⊢ Eq (f x) 0
  -/
  exact (hx.map f).eq_zero
  /-
    🎉 no goals
  -/


instance (ι) (R : ι → Type*) [∀ i, Zero (R i)] [∀ i, Pow (R i) ℕ]
    [∀ i, IsReduced (R i)] : IsReduced (∀ i, R i) where
  eq_zero _ := fun ⟨n, hn⟩ ↦ funext fun i ↦ IsReduced.eq_zero _ ⟨n, congr_fun hn i⟩


/-- An element `y` in a monoid is radical if for any element `x`, `y` divides `x` whenever it
  divides a power of `x`. -/
def IsRadical [Dvd R] [Pow R ℕ] (y : R) : Prop :=
  ∀ (n : ℕ) (x), y ∣ x ^ n → y ∣ x


theorem isRadical_iff_pow_one_lt [MonoidWithZero R] (k : ℕ) (hk : 1 < k) :
    IsRadical y ↔ ∀ x, y ∣ x ^ k → y ∣ x :=
  ⟨(· k), k.pow_imp_self_of_one_lt hk _ fun _ _ h ↦ .inl (dvd_mul_of_dvd_left h _)⟩


theorem isNilpotent_mul_left (h_comm : Commute x y) (h : IsNilpotent x) : IsNilpotent (x * y) := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    h : IsNilpotent x
    ⊢ IsNilpotent (HMul.hMul x y)
  -/
  obtain ⟨n, hn⟩ := h
  /-
    case intro
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    n : Nat
    hn : Eq (HPow.hPow x n) 0
    ⊢ IsNilpotent (HMul.hMul x y)
  -/
  use n
  /-
    case h
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    n : Nat
    hn : Eq (HPow.hPow x n) 0
    ⊢ Eq (HPow.hPow (HMul.hMul x y) n) 0
  -/
  rw [h_comm.mul_pow, hn, zero_mul]
  /-
    🎉 no goals
  -/


theorem isNilpotent_mul_right (h_comm : Commute x y) (h : IsNilpotent y) : IsNilpotent (x * y) := by
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    h : IsNilpotent y
    ⊢ IsNilpotent (HMul.hMul x y)
  -/
  rw [h_comm.eq]
  /-
    R : Type u_1
    x y : R
    inst✝ : Semiring R
    h_comm : Commute x y
    h : IsNilpotent y
    ⊢ IsNilpotent (HMul.hMul y x)
  -/
  exact h_comm.symm.isNilpotent_mul_left h
  /-
    🎉 no goals
  -/


