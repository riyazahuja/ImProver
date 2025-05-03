deriving instance Zero, CanonicallyOrderedCommSemiring, Nontrivial,
  LinearOrder, Bot, CanonicallyLinearOrderedAddCommMonoid, Sub,
  LinearOrderedAddCommMonoidWithTop, WellFoundedRelation
  for ENat
  -- AddCommMonoidWithOne,
  -- OrderBot, OrderTop, OrderedSub, SuccOrder, WellFoundedLt, CharZero

-- Porting Note: In `Data.Nat.ENatPart` proofs timed out when having
-- the `deriving AddCommMonoidWithOne`, and it seems to work without.


instance : OrderBot ℕ∞ := WithTop.orderBot

instance : OrderTop ℕ∞ := WithTop.orderTop

instance : OrderedSub ℕ∞ := inferInstanceAs (OrderedSub (WithTop ℕ))

instance : SuccOrder ℕ∞ := inferInstanceAs (SuccOrder (WithTop ℕ))

instance : WellFoundedLT ℕ∞ := inferInstanceAs (WellFoundedLT (WithTop ℕ))

instance : CharZero ℕ∞ := inferInstanceAs (CharZero (WithTop ℕ))


/-- Lemmas about `WithTop` expect (and can output) `WithTop.some` but the normal form for coercion
`ℕ → ℕ∞` is `Nat.cast`. -/
@[simp] theorem some_eq_coe : (WithTop.some : ℕ → ℕ∞) = Nat.cast := rfl


theorem coe_inj {a b : ℕ} : (a : ℕ∞) = b ↔ a = b := WithTop.coe_inj


instance : SuccAddOrder ℕ∞ where
                          /-
                            a b c m n x : ENat
                            ⊢ Eq (SuccOrder.succ x) (HAdd.hAdd x 1)
                          -/
                                      /-
                                        🎉 no goals
                                      -/
  succ_eq_add_one x := by cases x <;> simp [SuccOrder.succ]
                                      /-
                                        🎉 no goals
                                      -/

-- Porting note: `simp` and `norm_cast` can prove it
--@[simp, norm_cast]

theorem coe_zero : ((0 : ℕ) : ℕ∞) = 0 :=
  rfl

-- Porting note: `simp` and `norm_cast` can prove it
--@[simp, norm_cast]

theorem coe_one : ((1 : ℕ) : ℕ∞) = 1 :=
  rfl

-- Porting note: `simp` and `norm_cast` can prove it
--@[simp, norm_cast]

theorem coe_add (m n : ℕ) : ↑(m + n) = (m + n : ℕ∞) :=
  rfl


@[simp, norm_cast]
theorem coe_sub (m n : ℕ) : ↑(m - n) = (m - n : ℕ∞) :=
  rfl


@[simp] lemma coe_mul (m n : ℕ) : ↑(m * n) = (m * n : ℕ∞) := rfl


@[simp] theorem mul_top (hm : m ≠ 0) : m * ⊤ = ⊤ := WithTop.mul_top hm

@[simp] theorem top_mul (hm : m ≠ 0) : ⊤ * m = ⊤ := WithTop.top_mul hm


theorem top_pow {n : ℕ} (n_pos : 0 < n) : (⊤ : ℕ∞) ^ n = ⊤ := WithTop.top_pow n_pos


/-- Convert a `ℕ∞` to a `ℕ` using a proof that it is not infinite. -/
def lift (x : ℕ∞) (h : x < ⊤) : ℕ := WithTop.untop x (WithTop.lt_top_iff_ne_top.mp h)


@[simp] theorem coe_lift (x : ℕ∞) (h : x < ⊤) : (lift x h : ℕ∞) = x :=
  WithTop.coe_untop x (WithTop.lt_top_iff_ne_top.mp h)

@[simp] theorem lift_coe (n : ℕ) : lift (n : ℕ∞) (WithTop.coe_lt_top n) = n := rfl

@[simp] theorem lift_lt_iff {x : ℕ∞} {h} {n : ℕ} : lift x h < n ↔ x < n := WithTop.untop_lt_iff _

@[simp] theorem lift_le_iff {x : ℕ∞} {h} {n : ℕ} : lift x h ≤ n ↔ x ≤ n := WithTop.untop_le_iff _

@[simp] theorem lt_lift_iff {x : ℕ} {n : ℕ∞} {h} : x < lift n h ↔ x < n := WithTop.lt_untop_iff _

@[simp] theorem le_lift_iff {x : ℕ} {n : ℕ∞} {h} : x ≤ lift n h ↔ x ≤ n := WithTop.le_untop_iff _


@[simp] theorem lift_zero : lift 0 (WithTop.coe_lt_top 0) = 0 := rfl

@[simp] theorem lift_one : lift 1 (WithTop.coe_lt_top 1) = 1 := rfl

@[simp] theorem lift_ofNat (n : ℕ) [n.AtLeastTwo] :
    lift (no_index (OfNat.ofNat n)) (WithTop.coe_lt_top n) = OfNat.ofNat n := rfl


@[simp] theorem add_lt_top {a b : ℕ∞} : a + b < ⊤ ↔ a < ⊤ ∧ b < ⊤ := WithTop.add_lt_top


@[simp] theorem lift_add (a b : ℕ∞) (h : a + b < ⊤) :
    lift (a + b) h = lift a (add_lt_top.1 h).1 + lift b (add_lt_top.1 h).2 := by
  /-
    a b : ENat
    h : LT.lt (HAdd.hAdd a b) Top.top
    ⊢ Eq ((HAdd.hAdd a b).lift h) (HAdd.hAdd (a.lift ⋯) (b.lift ⋯))
  -/
  apply coe_inj.1
  /-
    a b : ENat
    h : LT.lt (HAdd.hAdd a b) Top.top
    ⊢ Eq ↑((HAdd.hAdd a b).lift h) ↑(HAdd.hAdd (a.lift ⋯) (b.lift ⋯))
  -/
  simp
  /-
    🎉 no goals
  -/


instance canLift : CanLift ℕ∞ ℕ (↑) (· ≠ ⊤) := WithTop.canLift


instance : WellFoundedRelation ℕ∞ where
  rel := (· < ·)
  wf := IsWellFounded.wf


/-- Conversion of `ℕ∞` to `ℕ` sending `∞` to `0`. -/
def toNat : ℕ∞ → ℕ := WithTop.untop' 0


/-- Homomorphism from `ℕ∞` to `ℕ` sending `∞` to `0`. -/
def toNatHom : MonoidWithZeroHom ℕ∞ ℕ where
  toFun := toNat
  map_one' := rfl
  map_zero' := rfl
  map_mul' := WithTop.untop'_zero_mul


@[simp, norm_cast] lemma coe_toNatHom : toNatHom = toNat := rfl


lemma toNatHom_apply (n : ℕ) : toNatHom n = toNat n := rfl


@[simp]
theorem toNat_coe (n : ℕ) : toNat n = n :=
  rfl


@[simp]
theorem toNat_zero : toNat 0 = 0 :=
  rfl


@[simp]
theorem toNat_one : toNat 1 = 1 :=
  rfl

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem toNat_ofNat (n : ℕ) [n.AtLeastTwo] : toNat (no_index (OfNat.ofNat n)) = n :=
  rfl


@[simp]
theorem toNat_top : toNat ⊤ = 0 :=
  rfl


@[simp] theorem toNat_eq_zero : toNat n = 0 ↔ n = 0 ∨ n = ⊤ := WithTop.untop'_eq_self_iff


@[simp]
theorem recTopCoe_zero {C : ℕ∞ → Sort*} (d : C ⊤) (f : ∀ a : ℕ, C a) : @recTopCoe C d f 0 = f 0 :=
  rfl


@[simp]
theorem recTopCoe_one {C : ℕ∞ → Sort*} (d : C ⊤) (f : ∀ a : ℕ, C a) : @recTopCoe C d f 1 = f 1 :=
  rfl

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem recTopCoe_ofNat {C : ℕ∞ → Sort*} (d : C ⊤) (f : ∀ a : ℕ, C a) (x : ℕ) [x.AtLeastTwo] :
    @recTopCoe C d f (no_index (OfNat.ofNat x)) = f (OfNat.ofNat x) :=
  rfl


@[simp]
theorem top_ne_coe (a : ℕ) : ⊤ ≠ (a : ℕ∞) :=
  nofun

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem top_ne_ofNat (a : ℕ) [a.AtLeastTwo] : ⊤ ≠ (no_index (OfNat.ofNat a : ℕ∞)) :=
  nofun


@[simp] lemma top_ne_zero : (⊤ : ℕ∞) ≠ 0 := nofun

@[simp] lemma top_ne_one : (⊤ : ℕ∞) ≠ 1 := nofun


@[simp]
theorem coe_ne_top (a : ℕ) : (a : ℕ∞) ≠ ⊤ :=
  nofun

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_ne_top (a : ℕ) [a.AtLeastTwo] : (no_index (OfNat.ofNat a : ℕ∞)) ≠ ⊤ :=
  nofun


@[simp] lemma zero_ne_top : 0 ≠ (⊤ : ℕ∞) := nofun

@[simp] lemma one_ne_top : 1 ≠ (⊤ : ℕ∞) := nofun


@[simp]
theorem top_sub_coe (a : ℕ) : (⊤ : ℕ∞) - a = ⊤ :=
  WithTop.top_sub_coe


@[simp]
theorem top_sub_one : (⊤ : ℕ∞) - 1 = ⊤ :=
  top_sub_coe 1

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem top_sub_ofNat (a : ℕ) [a.AtLeastTwo] : (⊤ : ℕ∞) - (no_index (OfNat.ofNat a)) = ⊤ :=
  top_sub_coe a


@[simp]
theorem top_pos : (0 : ℕ∞) < ⊤ :=
  WithTop.top_pos


@[deprecated ENat.top_pos (since := "2024-10-22")]
alias zero_lt_top := top_pos


theorem sub_top (a : ℕ∞) : a - ⊤ = 0 :=
  WithTop.sub_top


@[simp]
theorem coe_toNat_eq_self : ENat.toNat n = n ↔ n ≠ ⊤ :=
                     /-
                       n : ENat
                       ⊢ Iff (Eq (↑Top.top.toNat) Top.top) (Ne Top.top Top.top)
                     -/
                     /-
                       🎉 no goals
                     -/
  ENat.recTopCoe (by decide) (fun _ => by simp [toNat_coe]) n
                                          /-
                                            🎉 no goals
                                          -/


alias ⟨_, coe_toNat⟩ := coe_toNat_eq_self


theorem coe_toNat_le_self (n : ℕ∞) : ↑(toNat n) ≤ n :=
  ENat.recTopCoe le_top (fun _ => le_rfl) n


theorem toNat_add {m n : ℕ∞} (hm : m ≠ ⊤) (hn : n ≠ ⊤) : toNat (m + n) = toNat m + toNat n := by
  /-
    m n : ENat
    hm : Ne m Top.top
    hn : Ne n Top.top
    ⊢ Eq (HAdd.hAdd m n).toNat (HAdd.hAdd m.toNat n.toNat)
  -/
  lift m to ℕ using hm
  /-
    case intro
    n : ENat
    hn : Ne n Top.top
    m : Nat
    ⊢ Eq (HAdd.hAdd (↑m) n).toNat (HAdd.hAdd (↑m).toNat n.toNat)
  -/
  lift n to ℕ using hn
  /-
    case intro.intro
    m n : Nat
    ⊢ Eq (HAdd.hAdd ↑m ↑n).toNat (HAdd.hAdd (↑m).toNat (↑n).toNat)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toNat_sub {n : ℕ∞} (hn : n ≠ ⊤) (m : ℕ∞) : toNat (m - n) = toNat m - toNat n := by
  /-
    n : ENat
    hn : Ne n Top.top
    m : ENat
    ⊢ Eq (HSub.hSub m n).toNat (HSub.hSub m.toNat n.toNat)
  -/
  lift n to ℕ using hn
  /-
    case intro
    m : ENat
    n : Nat
    ⊢ Eq (HSub.hSub m ↑n).toNat (HSub.hSub m.toNat (↑n).toNat)
  -/
  induction m
    /-
      case intro.top
      n : Nat
      ⊢ Eq (HSub.hSub Top.top ↑n).toNat (HSub.hSub Top.top.toNat (↑n).toNat)
    -/
  · rw [top_sub_coe, toNat_top, zero_tsub]
    /-
      🎉 no goals
    -/
    /-
      case intro.coe
      n a✝ : Nat
      ⊢ Eq (HSub.hSub ↑a✝ ↑n).toNat (HSub.hSub (↑a✝).toNat (↑n).toNat)
    -/
  · rw [← coe_sub, toNat_coe, toNat_coe, toNat_coe]
    /-
      🎉 no goals
    -/


theorem toNat_eq_iff {m : ℕ∞} {n : ℕ} (hn : n ≠ 0) : toNat m = n ↔ m = n := by
  /-
    m : ENat
    n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq m.toNat n) (Eq m ↑n)
  -/
                  /-
                    🎉 no goals
                  -/
  induction m <;> simp [hn.symm]
                  /-
                    🎉 no goals
                  -/


lemma toNat_le_of_le_coe {m : ℕ∞} {n : ℕ} (h : m ≤ n) : toNat m ≤ n := by
  /-
    m : ENat
    n : Nat
    h : LE.le m ↑n
    ⊢ LE.le m.toNat n
  -/
  lift m to ℕ using ne_top_of_le_ne_top (coe_ne_top n) h
  /-
    case intro
    n m : Nat
    h : LE.le ↑m ↑n
    ⊢ LE.le (↑m).toNat n
  -/
  simpa using h
  /-
    🎉 no goals
  -/


@[gcongr]
lemma toNat_le_toNat {m n : ℕ∞} (h : m ≤ n) (hn : n ≠ ⊤) : toNat m ≤ toNat n :=
  toNat_le_of_le_coe <| h.trans_eq (coe_toNat hn).symm


@[simp]
theorem succ_def (m : ℕ∞) : Order.succ m = m + 1 :=
  Order.succ_eq_add_one m


@[deprecated Order.add_one_le_of_lt (since := "2024-09-04")]
theorem add_one_le_of_lt (h : m < n) : m + 1 ≤ n :=
  Order.add_one_le_of_lt h


theorem add_one_le_iff (hm : m ≠ ⊤) : m + 1 ≤ n ↔ m < n :=
  Order.add_one_le_iff_of_not_isMax (not_isMax_iff_ne_top.mpr hm)


@[deprecated Order.one_le_iff_pos (since := "2024-09-04")]
theorem one_le_iff_pos : 1 ≤ n ↔ 0 < n :=
  Order.one_le_iff_pos


theorem one_le_iff_ne_zero : 1 ≤ n ↔ n ≠ 0 :=
  Order.one_le_iff_pos.trans pos_iff_ne_zero


lemma lt_one_iff_eq_zero : n < 1 ↔ n = 0 :=
  not_le.symm.trans one_le_iff_ne_zero.not_left


@[deprecated Order.le_of_lt_add_one (since := "2024-09-04")]
theorem le_of_lt_add_one (h : m < n + 1) : m ≤ n :=
  Order.le_of_lt_add_one h


theorem lt_add_one_iff (hm : n ≠ ⊤) : m < n + 1 ↔ m ≤ n :=
  Order.lt_add_one_iff_of_not_isMax (not_isMax_iff_ne_top.mpr hm)


theorem le_coe_iff {n : ℕ∞} {k : ℕ} : n ≤ ↑k ↔ ∃ (n₀ : ℕ), n = n₀ ∧ n₀ ≤ k :=
  WithTop.le_coe_iff


@[simp]
lemma not_lt_zero (n : ℕ∞) : ¬ n < 0 := by
  /-
    n : ENat
    ⊢ Not (LT.lt n 0)
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


@[simp]
lemma coe_lt_top (n : ℕ) : (n : ℕ∞) < ⊤ :=
  WithTop.coe_lt_top n


                                                               /-
                                                                 n m : Nat
                                                                 ⊢ Iff (LT.lt ↑n ↑m) (LT.lt n m)
                                                               -/
lemma coe_lt_coe {n m : ℕ} : (n : ℕ∞) < (m : ℕ∞) ↔ n < m := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                               /-
                                                                 n m : Nat
                                                                 ⊢ Iff (LE.le ↑n ↑m) (LE.le n m)
                                                               -/
lemma coe_le_coe {n m : ℕ} : (n : ℕ∞) ≤ (m : ℕ∞) ↔ n ≤ m := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[elab_as_elim]
theorem nat_induction {P : ℕ∞ → Prop} (a : ℕ∞) (h0 : P 0) (hsuc : ∀ n : ℕ, P n → P n.succ)
    (htop : (∀ n : ℕ, P n) → P ⊤) : P a := by
  /-
    P : ENat → Prop
    a : ENat
    h0 : P 0
    hsuc : ∀ (n : Nat), P ↑n → P ↑n.succ
    htop : (∀ (n : Nat), P ↑n) → P Top.top
    ⊢ P a
  -/
  have A : ∀ n : ℕ, P n := fun n => Nat.recOn n h0 hsuc
  /-
    P : ENat → Prop
    a : ENat
    h0 : P 0
    hsuc : ∀ (n : Nat), P ↑n → P ↑n.succ
    htop : (∀ (n : Nat), P ↑n) → P Top.top
    A : ∀ (n : Nat), P ↑n
    ⊢ P a
  -/
  cases a
    /-
      case top
      P : ENat → Prop
      h0 : P 0
      hsuc : ∀ (n : Nat), P ↑n → P ↑n.succ
      htop : (∀ (n : Nat), P ↑n) → P Top.top
      A : ∀ (n : Nat), P ↑n
      ⊢ P Top.top
    -/
  · exact htop A
    /-
      🎉 no goals
    -/
    /-
      case coe
      P : ENat → Prop
      h0 : P 0
      hsuc : ∀ (n : Nat), P ↑n → P ↑n.succ
      htop : (∀ (n : Nat), P ↑n) → P Top.top
      A : ∀ (n : Nat), P ↑n
      a✝ : Nat
      ⊢ P ↑a✝
    -/
  · exact A _
    /-
      🎉 no goals
    -/


lemma add_one_pos : 0 < n + 1 :=
  succ_def n ▸ Order.bot_lt_succ n


lemma add_lt_add_iff_right {k : ℕ∞} (h : k ≠ ⊤) : n + k < m + k ↔ n < m :=
  WithTop.add_lt_add_iff_right h


lemma add_lt_add_iff_left {k : ℕ∞} (h : k ≠ ⊤) : k + n < k + m ↔ n < m :=
  WithTop.add_lt_add_iff_left h


protected lemma exists_nat_gt {n : ℕ∞} (hn : n ≠ ⊤) : ∃ m : ℕ, n < m := by
  /-
    n : ENat
    hn : Ne n Top.top
    ⊢ Exists fun m => LT.lt n ↑m
  -/
  lift n to ℕ using hn
  /-
    case intro
    n : Nat
    ⊢ Exists fun m => LT.lt ↑n ↑m
  -/
  obtain ⟨m, hm⟩ := exists_gt n
  /-
    case intro.intro
    n m : Nat
    hm : LT.lt n m
    ⊢ Exists fun m => LT.lt ↑n ↑m
  -/
  exact ⟨m, Nat.cast_lt.2 hm⟩
  /-
    🎉 no goals
  -/


lemma ne_top_iff_exists {x : ℕ∞} : x ≠ ⊤ ↔ ∃ a : ℕ, ↑a = x := WithTop.ne_top_iff_exists


@[simp] lemma sub_eq_top_iff : a - b = ⊤ ↔ a = ⊤ ∧ b ≠ ⊤ := WithTop.sub_eq_top_iff

lemma sub_ne_top_iff : a - b ≠ ⊤ ↔ a ≠ ⊤ ∨ b = ⊤ := WithTop.sub_ne_top_iff


lemma addLECancellable_of_ne_top : a ≠ ⊤ → AddLECancellable a := WithTop.addLECancellable_of_ne_top

lemma addLECancellable_of_lt_top : a < ⊤ → AddLECancellable a := WithTop.addLECancellable_of_lt_top


protected lemma le_sub_of_add_le_left (ha : a ≠ ⊤) : a + b ≤ c → b ≤ c - a :=
  (addLECancellable_of_ne_top ha).le_tsub_of_add_le_left


protected lemma sub_sub_cancel (h : a ≠ ⊤) (h2 : b ≤ a) : a - (a - b) = b :=
  (addLECancellable_of_ne_top <| ne_top_of_le_ne_top h tsub_le_self).tsub_tsub_cancel_of_le h2


lemma add_one_natCast_le_withTop_of_lt {m : ℕ} {n : WithTop ℕ∞} (h : m < n) : (m + 1 : ℕ) ≤ n := by
  match n with
  | ⊤ => exact le_top
  | (⊤ : ℕ∞) => exact WithTop.coe_le_coe.2 (OrderTop.le_top _)
  | (n : ℕ) => simpa only [Nat.cast_le, ge_iff_le, Nat.cast_lt] using h


@[simp] lemma coe_top_add_one : ((⊤ : ℕ∞) : WithTop ℕ∞) + 1 = (⊤ : ℕ∞) := rfl


@[simp] lemma add_one_eq_coe_top_iff {n : WithTop ℕ∞} : n + 1 = (⊤ : ℕ∞) ↔ n = (⊤ : ℕ∞) := by
  match n with
  | ⊤ => exact Iff.rfl
  | (⊤ : ℕ∞) => simp
  | (n : ℕ) =>
    norm_cast
    simp only [coe_ne_top, iff_false, ne_eq]


@[simp] lemma natCast_ne_coe_top (n : ℕ) : (n : WithTop ℕ∞) ≠ (⊤ : ℕ∞) := nofun


@[deprecated (since := "2024-10-22")]
alias nat_ne_coe_top := natCast_ne_coe_top


lemma one_le_iff_ne_zero_withTop {n : WithTop ℕ∞} : 1 ≤ n ↔ n ≠ 0 :=
  ⟨fun h ↦ (zero_lt_one.trans_le h).ne',
    fun h ↦ add_one_natCast_le_withTop_of_lt (pos_iff_ne_zero.mpr h)⟩


lemma natCast_le_of_coe_top_le_withTop {N : WithTop ℕ∞} (hN : (⊤ : ℕ∞) ≤ N) (n : ℕ) : n ≤ N :=
  le_trans (mod_cast le_top) hN


lemma natCast_lt_of_coe_top_le_withTop {N : WithTop ℕ∞} (hN : (⊤ : ℕ∞) ≤ N) (n : ℕ) : n < N :=
  lt_of_lt_of_le (mod_cast lt_add_one n) (natCast_le_of_coe_top_le_withTop hN (n + 1))


/--
Specialization of `WithTop.map` to `ENat`.
-/
def map (f : ℕ → α) (k : ℕ∞) : WithTop α := WithTop.map f k


@[simp]
theorem map_top (f : ℕ → α) : map f ⊤ = ⊤ := rfl


@[simp]
theorem map_coe (f : ℕ → α) (a : ℕ) : map f a = f a := rfl


@[simp]
theorem map_zero (f : ℕ → α) : map f 0 = f 0 := rfl


@[simp]
theorem map_one (f : ℕ → α) : map f 1 = f 1 := rfl


@[simp]
theorem map_ofNat (f : ℕ → α) (n : ℕ) [n.AtLeastTwo] : map f (no_index (OfNat.ofNat n)) = f n := rfl


@[simp]
lemma map_eq_top_iff {f : ℕ → α} : map f n = ⊤ ↔ n = ⊤ := WithTop.map_eq_top_iff


@[simp]
theorem map_natCast_nonneg [AddMonoidWithOne α] [PartialOrder α]
    [AddLeftMono α] [ZeroLEOneClass α] : 0 ≤ n.map (Nat.cast : ℕ → α) := by
  /-
    n : ENat
    α : Type u_1
    inst✝³ : AddMonoidWithOne α
    inst✝² : PartialOrder α
    inst✝¹ : AddLeftMono α
    inst✝ : ZeroLEOneClass α
    ⊢ LE.le 0 (ENat.map Nat.cast n)
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp
              /-
                🎉 no goals
              -/


@[simp]
theorem strictMono_map_iff {f : ℕ → α} [Preorder α] : StrictMono (ENat.map f) ↔ StrictMono f :=
  WithTop.strictMono_map_iff


@[simp]
theorem monotone_map_iff {f : ℕ → α} [Preorder α] : Monotone (ENat.map f) ↔ Monotone f :=
  WithTop.monotone_map_iff


@[simp]
protected theorem map_add {β F} [Add β] [FunLike F ℕ β] [AddHomClass F ℕ β]
    (f : F) (a b : ℕ∞) : (a + b).map f = a.map f + b.map f :=
  WithTop.map_add f a b


/-- A version of `ENat.map` for `OneHom`s. -/
-- @[to_additive (attr := simps (config := .asFn))
--   "A version of `ENat.map` for `ZeroHom`s"]
protected def _root_.OneHom.ENatMap {N : Type*} [One N] (f : OneHom ℕ N) :
    OneHom ℕ∞ (WithTop N) where
  toFun := ENat.map f
                 /-
                   a b c m n : ENat
                   α : Type u_1
                   N : Type u_2
                   inst✝ : One N
                   f : OneHom Nat N
                   ⊢ Eq (ENat.map (⇑f) 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/


/-- A version of `ENat.map` for `ZeroHom`s. -/
protected def _root_.ZeroHom.ENatMap {N : Type*} [Zero N] (f : ZeroHom ℕ N) :
    ZeroHom ℕ∞ (WithTop N) where
  toFun := ENat.map f
                  /-
                    a b c m n : ENat
                    α : Type u_1
                    N : Type u_2
                    inst✝ : Zero N
                    f : ZeroHom Nat N
                    ⊢ Eq (ENat.map (⇑f) 0) 0
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/


/-- A version of `WithTop.map` for `AddHom`s. -/
@[simps (config := .asFn)]
protected def _root_.AddHom.ENatMap {N : Type*} [Add N] (f : AddHom ℕ N) :
    AddHom ℕ∞ (WithTop N) where
  toFun := ENat.map f
  map_add' := ENat.map_add f


/-- A version of `WithTop.map` for `AddMonoidHom`s. -/
@[simps (config := .asFn)]
protected def _root_.AddMonoidHom.ENatMap {N : Type*} [AddZeroClass N]
    (f : ℕ →+ N) : ℕ∞ →+ WithTop N :=
  { ZeroHom.ENatMap f.toZeroHom, AddHom.ENatMap f.toAddHom with toFun := ENat.map f }


/-- A version of `ENat.map` for `MonoidWithZeroHom`s. -/
@[simps (config := .asFn)]
protected def _root_.MonoidWithZeroHom.ENatMap {S : Type*} [MulZeroOneClass S] [DecidableEq S]
    [Nontrivial S] (f : ℕ →*₀ S)
    (hf : Function.Injective f) : ℕ∞ →*₀ WithTop S :=
  { f.toZeroHom.ENatMap, f.toMonoidHom.toOneHom.ENatMap with
    toFun := ENat.map f
    map_mul' := fun x y => by
      have : ∀ z, map f z = 0 ↔ z = 0 := fun z =>
        (Option.map_injective hf).eq_iff' f.toZeroHom.ENatMap.map_zero
      /-
        a b c m n : ENat
        α : Type u_1
        S : Type u_2
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom Nat S
        hf : Function.Injective ⇑f
        x y : ENat
        this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
        ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x y)) (HMul.hM …
      -/
      rcases Decidable.eq_or_ne x 0 with (rfl | hx)
        /-
          case inl
          a b c m n : ENat
          α : Type u_1
          S : Type u_2
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom Nat S
          hf : Function.Injective ⇑f
          y : ENat
          this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
          ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul 0 y)) (HMul.hM …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inr
        a b c m n : ENat
        α : Type u_1
        S : Type u_2
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom Nat S
        hf : Function.Injective ⇑f
        x y : ENat
        this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
        hx : Ne x 0
        ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x y)) (HMul.hM …
      -/
      rcases Decidable.eq_or_ne y 0 with (rfl | hy)
        /-
          case inr.inl
          a b c m n : ENat
          α : Type u_1
          S : Type u_2
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom Nat S
          hf : Function.Injective ⇑f
          x : ENat
          this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
          hx : Ne x 0
          ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x 0)) (HMul.hM …
        -/
      · simp
        /-
          🎉 no goals
        -/
      /-
        case inr.inr
        a b c m n : ENat
        α : Type u_1
        S : Type u_2
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom Nat S
        hf : Function.Injective ⇑f
        x y : ENat
        this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
        hx : Ne x 0
        hy : Ne y 0
        ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul x y)) (HMul.hM …
      -/
      induction' x with x
        /-
          case inr.inr.top
          a b c m n : ENat
          α : Type u_1
          S : Type u_2
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom Nat S
          hf : Function.Injective ⇑f
          y : ENat
          this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
          hy : Ne y 0
          hx : Ne Top.top 0
          ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul Top.top y)) (H …
        -/
      · simp [hy, this]
        /-
          🎉 no goals
        -/
      /-
        case inr.inr.coe
        a b c m n : ENat
        α : Type u_1
        S : Type u_2
        inst✝² : MulZeroOneClass S
        inst✝¹ : DecidableEq S
        inst✝ : Nontrivial S
        f : MonoidWithZeroHom Nat S
        hf : Function.Injective ⇑f
        y : ENat
        this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
        hy : Ne y 0
        x : Nat
        hx : Ne (↑x) 0
        ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul (↑x) y)) (HMul …
      -/
      induction' y with y
        /-
          case inr.inr.coe.top
          a b c m n : ENat
          α : Type u_1
          S : Type u_2
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom Nat S
          hf : Function.Injective ⇑f
          this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
          x : Nat
          hx : Ne (↑x) 0
          hy : Ne Top.top 0
          ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul (↑x) Top.top)) …
        -/
      · have : (f x : WithTop S) ≠ 0 := by simpa [hf.eq_iff' (_root_.map_zero f)] using hx
        /-
          case inr.inr.coe.top
          a b c m n : ENat
          α : Type u_1
          S : Type u_2
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom Nat S
          hf : Function.Injective ⇑f
          this✝ : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
          x : Nat
          hx : Ne (↑x) 0
          hy : Ne Top.top 0
          this : Ne (↑(f x)) 0
          ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul (↑x) Top.top)) …
        -/
        simp [mul_top hx, WithTop.mul_top this]
        /-
          🎉 no goals
        -/
      · -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: `simp [← coe_mul]` times out
        /-
          case inr.inr.coe.coe
          a b c m n : ENat
          α : Type u_1
          S : Type u_2
          inst✝² : MulZeroOneClass S
          inst✝¹ : DecidableEq S
          inst✝ : Nontrivial S
          f : MonoidWithZeroHom Nat S
          hf : Function.Injective ⇑f
          this : ∀ (z : ENat), Iff (Eq (ENat.map (⇑f) z) 0) (Eq z 0)
          x : Nat
          hx : Ne (↑x) 0
          y : Nat
          hy : Ne (↑y) 0
          ⊢ Eq ({ toFun := ENat.map ⇑f, map_zero' := ⋯ }.toFun (HMul.hMul ↑x ↑y)) (HMul. …
        -/
        simp only [map_coe, ← coe_mul, map_mul, WithTop.coe_mul] }
        /-
          🎉 no goals
        -/


/-- A version of `ENat.map` for `RingHom`s. -/
@[simps (config := .asFn)]
protected def _root_.RingHom.ENatMap {S : Type*} [CanonicallyOrderedCommSemiring S] [DecidableEq S]
    [Nontrivial S] (f : ℕ →+* S) (hf : Function.Injective f) : ℕ∞ →+* WithTop S :=
  {MonoidWithZeroHom.ENatMap f.toMonoidWithZeroHom hf, f.toAddMonoidHom.ENatMap with}


lemma WithBot.lt_add_one_iff {n : WithBot ℕ∞} {m : ℕ} : n < m + 1 ↔ n ≤ m := by
  /-
    n : WithBot ENat
    m : Nat
    ⊢ Iff (LT.lt n (HAdd.hAdd (↑m) 1)) (LE.le n ↑m)
  -/
  rw [← WithBot.coe_one, ← ENat.coe_one, WithBot.coe_natCast, ← Nat.cast_add, ← WithBot.coe_natCast]
  /-
    n : WithBot ENat
    m : Nat
    ⊢ Iff (LT.lt n ↑↑(HAdd.hAdd m 1)) (LE.le n ↑m)
  -/
  cases n
    /-
      case bot
      m : Nat
      ⊢ Iff (LT.lt Bot.bot ↑↑(HAdd.hAdd m 1)) (LE.le Bot.bot ↑m)
    -/
  · simp only [bot_le, iff_true, WithBot.bot_lt_coe]
    /-
      🎉 no goals
    -/
  · rw [WithBot.coe_lt_coe, Nat.cast_add, ENat.coe_one, ENat.lt_add_one_iff (ENat.coe_ne_top _),
      ← WithBot.coe_le_coe, WithBot.coe_natCast]


lemma WithBot.add_one_le_iff {n : ℕ} {m : WithBot ℕ∞} : n + 1 ≤ m ↔ n < m := by
  /-
    n : Nat
    m : WithBot ENat
    ⊢ Iff (LE.le (HAdd.hAdd (↑n) 1) m) (LT.lt (↑n) m)
  -/
  rw [← WithBot.coe_one, ← ENat.coe_one, WithBot.coe_natCast, ← Nat.cast_add, ← WithBot.coe_natCast]
  /-
    n : Nat
    m : WithBot ENat
    ⊢ Iff (LE.le (↑↑(HAdd.hAdd n 1)) m) (LT.lt (↑n) m)
  -/
  cases m
    /-
      case bot
      n : Nat
      ⊢ Iff (LE.le (↑↑(HAdd.hAdd n 1)) Bot.bot) (LT.lt (↑n) Bot.bot)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [WithBot.coe_le_coe, ENat.coe_add, ENat.coe_one, ENat.add_one_le_iff (ENat.coe_ne_top n),
      ← WithBot.coe_lt_coe, WithBot.coe_natCast]

