deriving instance AddLeftCancelSemigroup, AddRightCancelSemigroup, AddCommSemigroup,
  LinearOrderedCancelCommMonoid, Add, Mul, Distrib for PNat


instance instWellFoundedLT : WellFoundedLT ℕ+ := WellFoundedRelation.isWellFounded


@[simp]
theorem one_add_natPred (n : ℕ+) : 1 + n.natPred = n := by
  /-
    n : PNat
    ⊢ Eq (HAdd.hAdd 1 n.natPred) ↑n
  -/
  rw [natPred, add_tsub_cancel_iff_le.mpr <| show 1 ≤ (n : ℕ) from n.2]
  /-
    🎉 no goals
  -/


@[simp]
theorem natPred_add_one (n : ℕ+) : n.natPred + 1 = n :=
  (add_comm _ _).trans n.one_add_natPred


@[mono]
theorem natPred_strictMono : StrictMono natPred := fun m _ h => Nat.pred_lt_pred m.2.ne' h


@[mono]
theorem natPred_monotone : Monotone natPred :=
  natPred_strictMono.monotone


theorem natPred_injective : Function.Injective natPred :=
  natPred_strictMono.injective


@[simp]
theorem natPred_lt_natPred {m n : ℕ+} : m.natPred < n.natPred ↔ m < n :=
  natPred_strictMono.lt_iff_lt


@[simp]
theorem natPred_le_natPred {m n : ℕ+} : m.natPred ≤ n.natPred ↔ m ≤ n :=
  natPred_strictMono.le_iff_le


@[simp]
theorem natPred_inj {m n : ℕ+} : m.natPred = n.natPred ↔ m = n :=
  natPred_injective.eq_iff


@[simp, norm_cast]
lemma val_ofNat (n : ℕ) [NeZero n] :
    ((no_index (OfNat.ofNat n) : ℕ+) : ℕ) = OfNat.ofNat n :=
  rfl


@[simp]
lemma mk_ofNat (n : ℕ) (h : 0 < n) :
    @Eq ℕ+ (⟨no_index (OfNat.ofNat n), h⟩ : ℕ+) (haveI : NeZero n := ⟨h.ne'⟩; OfNat.ofNat n) :=
  rfl


@[mono]
theorem succPNat_strictMono : StrictMono succPNat := fun _ _ => Nat.succ_lt_succ


@[mono]
theorem succPNat_mono : Monotone succPNat :=
  succPNat_strictMono.monotone


@[simp]
theorem succPNat_lt_succPNat {m n : ℕ} : m.succPNat < n.succPNat ↔ m < n :=
  succPNat_strictMono.lt_iff_lt


@[simp]
theorem succPNat_le_succPNat {m n : ℕ} : m.succPNat ≤ n.succPNat ↔ m ≤ n :=
  succPNat_strictMono.le_iff_le


theorem succPNat_injective : Function.Injective succPNat :=
  succPNat_strictMono.injective


@[simp]
theorem succPNat_inj {n m : ℕ} : succPNat n = succPNat m ↔ n = m :=
  succPNat_injective.eq_iff


/-- We now define a long list of structures on `ℕ+` induced by
 similar structures on `ℕ`. Most of these behave in a completely
 obvious way, but there are a few things to be said about
 subtraction, division and powers.
-/
@[simp, norm_cast]
theorem coe_inj {m n : ℕ+} : (m : ℕ) = n ↔ m = n :=
  SetCoe.ext_iff


@[simp, norm_cast]
theorem add_coe (m n : ℕ+) : ((m + n : ℕ+) : ℕ) = m + n :=
  rfl


/-- `coe` promoted to an `AddHom`, that is, a morphism which preserves addition. -/
def coeAddHom : AddHom ℕ+ ℕ where
  toFun := Coe.coe
  map_add' := add_coe


instance addLeftMono : AddLeftMono ℕ+ :=
  Positive.addLeftMono


instance addLeftStrictMono : AddLeftStrictMono ℕ+ :=
  Positive.addLeftStrictMono


instance addLeftReflectLE : AddLeftReflectLE ℕ+ :=
  Positive.addLeftReflectLE


instance addLeftReflectLT : AddLeftReflectLT ℕ+ :=
  Positive.addLeftReflectLT


/-- The order isomorphism between ℕ and ℕ+ given by `succ`. -/
@[simps! (config := .asFn) apply]
def _root_.OrderIso.pnatIsoNat : ℕ+ ≃o ℕ where
  toEquiv := Equiv.pnatEquivNat
  map_rel_iff' := natPred_le_natPred


@[simp]
theorem _root_.OrderIso.pnatIsoNat_symm_apply : OrderIso.pnatIsoNat.symm = Nat.succPNat :=
  rfl


theorem lt_add_one_iff : ∀ {a b : ℕ+}, a < b + 1 ↔ a ≤ b := Nat.lt_add_one_iff


theorem add_one_le_iff : ∀ {a b : ℕ+}, a + 1 ≤ b ↔ a < b := Nat.add_one_le_iff


instance instOrderBot : OrderBot ℕ+ where
  bot := 1
  bot_le a := a.property


@[simp]
theorem bot_eq_one : (⊥ : ℕ+) = 1 :=
  rfl


/-- Strong induction on `ℕ+`, with `n = 1` treated separately. -/
def caseStrongInductionOn {p : ℕ+ → Sort*} (a : ℕ+) (hz : p 1)
    (hi : ∀ n, (∀ m, m ≤ n → p m) → p (n + 1)) : p a := by
  /-
    p : PNat → Sort u_1
    a : PNat
    hz : p 1
    hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
    ⊢ p a
  -/
  apply strongInductionOn a
  /-
    p : PNat → Sort u_1
    a : PNat
    hz : p 1
    hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
    ⊢ (k : PNat) → ((m : PNat) → LT.lt m k → p m) → p k
  -/
  rintro ⟨k, kprop⟩ hk
  /-
    case mk
    p : PNat → Sort u_1
    a : PNat
    hz : p 1
    hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
    k : Nat
    kprop : LT.lt 0 k
    hk : (m : PNat) → LT.lt m ⟨k, kprop⟩ → p m
    ⊢ p ⟨k, kprop⟩
  -/
  cases' k with k
    /-
      case mk.zero
      p : PNat → Sort u_1
      a : PNat
      hz : p 1
      hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
      kprop : LT.lt 0 0
      hk : (m : PNat) → LT.lt m ⟨0, kprop⟩ → p m
      ⊢ p ⟨0, kprop⟩
    -/
  · exact (lt_irrefl 0 kprop).elim
    /-
      🎉 no goals
    -/
  /-
    case mk.succ
    p : PNat → Sort u_1
    a : PNat
    hz : p 1
    hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
    k : Nat
    kprop : LT.lt 0 (HAdd.hAdd k 1)
    hk : (m : PNat) → LT.lt m ⟨HAdd.hAdd k 1, kprop⟩ → p m
    ⊢ p ⟨HAdd.hAdd k 1, kprop⟩
  -/
  cases' k with k
    /-
      case mk.succ.zero
      p : PNat → Sort u_1
      a : PNat
      hz : p 1
      hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
      kprop : LT.lt 0 (HAdd.hAdd 0 1)
      hk : (m : PNat) → LT.lt m ⟨HAdd.hAdd 0 1, kprop⟩ → p m
      ⊢ p ⟨HAdd.hAdd 0 1, kprop⟩
    -/
  · exact hz
    /-
      🎉 no goals
    -/
  /-
    case mk.succ.succ
    p : PNat → Sort u_1
    a : PNat
    hz : p 1
    hi : (n : PNat) → ((m : PNat) → LE.le m n → p m) → p (HAdd.hAdd n 1)
    k : Nat
    kprop : LT.lt 0 (HAdd.hAdd (HAdd.hAdd k 1) 1)
    hk : (m : PNat) → LT.lt m ⟨HAdd.hAdd (HAdd.hAdd k 1) 1, kprop⟩ → p m
    ⊢ p ⟨HAdd.hAdd (HAdd.hAdd k 1) 1, kprop⟩
  -/
  exact hi ⟨k.succ, Nat.succ_pos _⟩ fun m hm => hk _ (Nat.lt_succ_iff.2 hm)
  /-
    🎉 no goals
  -/


/-- An induction principle for `ℕ+`: it takes values in `Sort*`, so it applies also to Types,
not only to `Prop`. -/
@[elab_as_elim]
def recOn (n : ℕ+) {p : ℕ+ → Sort*} (p1 : p 1) (hp : ∀ n, p n → p (n + 1)) : p n := by
  /-
    n : PNat
    p : PNat → Sort u_1
    p1 : p 1
    hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
    ⊢ p n
  -/
  rcases n with ⟨n, h⟩
  /-
    case mk
    p : PNat → Sort u_1
    p1 : p 1
    hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
    n : Nat
    h : LT.lt 0 n
    ⊢ p ⟨n, h⟩
  -/
  induction' n with n IH
    /-
      case mk.zero
      p : PNat → Sort u_1
      p1 : p 1
      hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
      h : LT.lt 0 0
      ⊢ p ⟨0, h⟩
    -/
  · exact absurd h (by decide)
    /-
      🎉 no goals
    -/
    /-
      case mk.succ
      p : PNat → Sort u_1
      p1 : p 1
      hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
      n : Nat
      IH : (h : LT.lt 0 n) → p ⟨n, h⟩
      h : LT.lt 0 (HAdd.hAdd n 1)
      ⊢ p ⟨HAdd.hAdd n 1, h⟩
    -/
  · cases' n with n
      /-
        case mk.succ.zero
        p : PNat → Sort u_1
        p1 : p 1
        hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
        IH : (h : LT.lt 0 0) → p ⟨0, h⟩
        h : LT.lt 0 (HAdd.hAdd 0 1)
        ⊢ p ⟨HAdd.hAdd 0 1, h⟩
      -/
    · exact p1
      /-
        🎉 no goals
      -/
      /-
        case mk.succ.succ
        p : PNat → Sort u_1
        p1 : p 1
        hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
        n : Nat
        IH : (h : LT.lt 0 (HAdd.hAdd n 1)) → p ⟨HAdd.hAdd n 1, h⟩
        h : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ p ⟨HAdd.hAdd (HAdd.hAdd n 1) 1, h⟩
      -/
    · exact hp _ (IH n.succ_pos)
      /-
        🎉 no goals
      -/


@[simp]
theorem recOn_one {p} (p1 hp) : @PNat.recOn 1 p p1 hp = p1 :=
  rfl


@[simp]
theorem recOn_succ (n : ℕ+) {p : ℕ+ → Sort*} (p1 hp) :
    @PNat.recOn (n + 1) p p1 hp = hp n (@PNat.recOn n p p1 hp) := by
  /-
    n : PNat
    p : PNat → Sort u_1
    p1 : p 1
    hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
    ⊢ Eq ((HAdd.hAdd n 1).recOn p1 hp) (hp n (n.recOn p1 hp))
  -/
  cases' n with n h
  /-
    case mk
    p : PNat → Sort u_1
    p1 : p 1
    hp : (n : PNat) → p n → p (HAdd.hAdd n 1)
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq ((HAdd.hAdd ⟨n, h⟩ 1).recOn p1 hp) (hp ⟨n, h⟩ (PNat.recOn ⟨n, h⟩ p1 hp))
  -/
  cases n <;> [exact absurd h (by decide); rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofNat_le_ofNat {m n : ℕ} [NeZero m] [NeZero n] :
    (no_index (OfNat.ofNat m) : ℕ+) ≤ no_index (OfNat.ofNat n) ↔ OfNat.ofNat m ≤ OfNat.ofNat n :=
  .rfl


@[simp]
theorem ofNat_lt_ofNat {m n : ℕ} [NeZero m] [NeZero n] :
    (no_index (OfNat.ofNat m) : ℕ+) < no_index (OfNat.ofNat n) ↔ OfNat.ofNat m < OfNat.ofNat n :=
  .rfl


@[simp]
theorem ofNat_inj {m n : ℕ} [NeZero m] [NeZero n] :
    (no_index (OfNat.ofNat m) : ℕ+) = no_index (OfNat.ofNat n) ↔ OfNat.ofNat m = OfNat.ofNat n :=
  Subtype.mk_eq_mk


@[simp, norm_cast]
theorem mul_coe (m n : ℕ+) : ((m * n : ℕ+) : ℕ) = m * n :=
  rfl


/-- `PNat.coe` promoted to a `MonoidHom`. -/
def coeMonoidHom : ℕ+ →* ℕ where
  toFun := Coe.coe
  map_one' := one_coe
  map_mul' := mul_coe


@[simp]
theorem coe_coeMonoidHom : (coeMonoidHom : ℕ+ → ℕ) = Coe.coe :=
  rfl


@[simp]
theorem le_one_iff {n : ℕ+} : n ≤ 1 ↔ n = 1 :=
  le_bot_iff


theorem lt_add_left (n m : ℕ+) : n < m + n :=
  lt_add_of_pos_left _ m.2


theorem lt_add_right (n m : ℕ+) : n < n + m :=
  (lt_add_left n m).trans_eq (add_comm _ _)


@[simp, norm_cast]
theorem pow_coe (m : ℕ+) (n : ℕ) : ↑(m ^ n) = (m : ℕ) ^ n :=
  rfl


/-- b is greater one if any a is less than b -/
theorem one_lt_of_lt {a b : ℕ+} (hab : a < b) : 1 < b := bot_le.trans_lt hab


theorem add_one (a : ℕ+) : a + 1 = succPNat a := rfl


theorem lt_succ_self (a : ℕ+) : a < succPNat a := lt.base a


/-- Subtraction a - b is defined in the obvious way when
  a > b, and by a - b = 1 if a ≤ b.
-/
instance instSub : Sub ℕ+ :=
  ⟨fun a b => toPNat' (a - b : ℕ)⟩


theorem sub_coe (a b : ℕ+) : ((a - b : ℕ+) : ℕ) = ite (b < a) (a - b : ℕ) 1 := by
  /-
    a b : PNat
    ⊢ Eq (↑(HSub.hSub a b)) (ite (LT.lt b a) (HSub.hSub ↑a ↑b) 1)
  -/
  change (toPNat' _ : ℕ) = ite _ _ _
  /-
    a b : PNat
    ⊢ Eq (↑(HSub.hSub ↑a ↑b).toPNat') (ite (LT.lt b a) (HSub.hSub ↑a ↑b) 1)
  -/
  split_ifs with h
    /-
      case pos
      a b : PNat
      h : LT.lt b a
      ⊢ Eq (↑(HSub.hSub ↑a ↑b).toPNat') (HSub.hSub ↑a ↑b)
    -/
  · exact toPNat'_coe (tsub_pos_of_lt h)
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b : PNat
      h : Not (LT.lt b a)
      ⊢ Eq (↑(HSub.hSub ↑a ↑b).toPNat') 1
    -/
  · rw [tsub_eq_zero_iff_le.mpr (le_of_not_gt h : (a : ℕ) ≤ b)]
    /-
      case neg
      a b : PNat
      h : Not (LT.lt b a)
      ⊢ Eq (↑(Nat.toPNat' 0)) 1
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem sub_le (a b : ℕ+) : a - b ≤ a := by
  /-
    a b : PNat
    ⊢ LE.le (HSub.hSub a b) a
  -/
  rw [← coe_le_coe, sub_coe]
  /-
    a b : PNat
    ⊢ LE.le (ite (LT.lt b a) (HSub.hSub ↑a ↑b) 1) ↑a
  -/
  split_ifs with h
    /-
      case pos
      a b : PNat
      h : LT.lt b a
      ⊢ LE.le (HSub.hSub ↑a ↑b) ↑a
    -/
  · exact Nat.sub_le a b
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b : PNat
      h : Not (LT.lt b a)
      ⊢ LE.le 1 ↑a
    -/
  · exact a.2
    /-
      🎉 no goals
    -/


theorem le_sub_one_of_lt {a b : ℕ+} (hab : a < b) : a ≤ b - (1 : ℕ+) := by
  /-
    a b : PNat
    hab : LT.lt a b
    ⊢ LE.le a (HSub.hSub b 1)
  -/
  rw [← coe_le_coe, sub_coe]
  /-
    a b : PNat
    hab : LT.lt a b
    ⊢ LE.le (↑a) (ite (LT.lt 1 b) (HSub.hSub ↑b ↑1) 1)
  -/
  split_ifs with h
    /-
      case pos
      a b : PNat
      hab : LT.lt a b
      h : LT.lt 1 b
      ⊢ LE.le (↑a) (HSub.hSub ↑b ↑1)
    -/
  · exact Nat.le_pred_of_lt hab
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b : PNat
      hab : LT.lt a b
      h : Not (LT.lt 1 b)
      ⊢ LE.le (↑a) 1
    -/
  · exact hab.le.trans (le_of_not_lt h)
    /-
      🎉 no goals
    -/


theorem add_sub_of_lt {a b : ℕ+} : a < b → a + (b - a) = b :=
  fun h =>
    PNat.eq <| by
      /-
        a b : PNat
        h : LT.lt a b
        ⊢ Eq ↑(HAdd.hAdd a (HSub.hSub b a)) ↑b
      -/
      rw [add_coe, sub_coe, if_pos h]
      /-
        a b : PNat
        h : LT.lt a b
        ⊢ Eq (HAdd.hAdd (↑a) (HSub.hSub ↑b ↑a)) ↑b
      -/
      exact add_tsub_cancel_of_le h.le
      /-
        🎉 no goals
      -/


theorem sub_add_of_lt {a b : ℕ+} (h : b < a) : a - b + b = a := by
  /-
    a b : PNat
    h : LT.lt b a
    ⊢ Eq (HAdd.hAdd (HSub.hSub a b) b) a
  -/
  rw [add_comm, add_sub_of_lt h]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_sub {a b : ℕ+} : a + b - b = a :=
  add_right_cancel (sub_add_of_lt (lt_add_left _ _))


/-- If `n : ℕ+` is different from `1`, then it is the successor of some `k : ℕ+`. -/
theorem exists_eq_succ_of_ne_one : ∀ {n : ℕ+} (_ : n ≠ 1), ∃ k : ℕ+, n = k + 1
  | ⟨1, _⟩, h₁ => False.elim <| h₁ rfl
                                 /-
                                   n : Nat
                                   property✝ : LT.lt 0 (HAdd.hAdd n 2)
                                   x✝ : Ne ⟨HAdd.hAdd n 2, property✝⟩ 1
                                   ⊢ LT.lt 0 (HAdd.hAdd n 1)
                                 -/
  | ⟨n + 2, _⟩, _ => ⟨⟨n + 1, by simp⟩, rfl⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- Lemmas with div, dvd and mod operations -/
theorem modDivAux_spec :
    ∀ (k : ℕ+) (r q : ℕ) (_ : ¬(r = 0 ∧ q = 0)),
      ((modDivAux k r q).1 : ℕ) + k * (modDivAux k r q).2 = r + k * q
  | _, 0, 0, h => (h ⟨rfl, rfl⟩).elim
  | k, 0, q + 1, _ => by
    /-
      k : PNat
      q : Nat
      x✝ : Not (And (Eq 0 0) (Eq (HAdd.hAdd q 1) 0))
      ⊢ Eq (HAdd.hAdd (↑(k.modDivAux 0 (HAdd.hAdd q 1)).1) (HMul.hMul (↑k) (k.modDiv …
    -/
    change (k : ℕ) + (k : ℕ) * (q + 1).pred = 0 + (k : ℕ) * (q + 1)
    /-
      k : PNat
      q : Nat
      x✝ : Not (And (Eq 0 0) (Eq (HAdd.hAdd q 1) 0))
      ⊢ Eq (HAdd.hAdd (↑k) (HMul.hMul (↑k) (HAdd.hAdd q 1).pred)) (HAdd.hAdd 0 (HMul …
    -/
    rw [Nat.pred_succ, Nat.mul_succ, zero_add, add_comm]
    /-
      🎉 no goals
    -/
  | _, _ + 1, _, _ => rfl


theorem mod_add_div (m k : ℕ+) : (mod m k + k * div m k : ℕ) = m := by
  /-
    m k : PNat
    ⊢ Eq (HAdd.hAdd (↑(m.mod k)) (HMul.hMul (↑k) (m.div k))) ↑m
  -/
  let h₀ := Nat.mod_add_div (m : ℕ) (k : ℕ)
  have : ¬((m : ℕ) % (k : ℕ) = 0 ∧ (m : ℕ) / (k : ℕ) = 0) := by
    rintro ⟨hr, hq⟩
    rw [hr, hq, mul_zero, zero_add] at h₀
    exact (m.ne_zero h₀.symm).elim
  /-
    m k : PNat
    h₀ : Eq (HAdd.hAdd (HMod.hMod ↑m ↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) ↑m := …
    this : Not (And (Eq (HMod.hMod ↑m ↑k) 0) (Eq (HDiv.hDiv ↑m ↑k) 0))
    ⊢ Eq (HAdd.hAdd (↑(m.mod k)) (HMul.hMul (↑k) (m.div k))) ↑m
  -/
  have := modDivAux_spec k ((m : ℕ) % (k : ℕ)) ((m : ℕ) / (k : ℕ)) this
  /-
    m k : PNat
    h₀ : Eq (HAdd.hAdd (HMod.hMod ↑m ↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) ↑m := …
    this✝ : Not (And (Eq (HMod.hMod ↑m ↑k) 0) (Eq (HDiv.hDiv ↑m ↑k) 0))
    this : Eq (HAdd.hAdd (↑(k.modDivAux (HMod.hMod ↑m ↑k) (HDiv.hDiv ↑m ↑k)).1) (H …
    ⊢ Eq (HAdd.hAdd (↑(m.mod k)) (HMul.hMul (↑k) (m.div k))) ↑m
  -/
  exact this.trans h₀
  /-
    🎉 no goals
  -/


theorem div_add_mod (m k : ℕ+) : (k * div m k + mod m k : ℕ) = m :=
  (add_comm _ _).trans (mod_add_div _ _)


theorem mod_add_div' (m k : ℕ+) : (mod m k + div m k * k : ℕ) = m := by
  /-
    m k : PNat
    ⊢ Eq (HAdd.hAdd (↑(m.mod k)) (HMul.hMul (m.div k) ↑k)) ↑m
  -/
  rw [mul_comm]
  /-
    m k : PNat
    ⊢ Eq (HAdd.hAdd (↑(m.mod k)) (HMul.hMul (↑k) (m.div k))) ↑m
  -/
  exact mod_add_div _ _
  /-
    🎉 no goals
  -/


theorem div_add_mod' (m k : ℕ+) : (div m k * k + mod m k : ℕ) = m := by
  /-
    m k : PNat
    ⊢ Eq (HAdd.hAdd (HMul.hMul (m.div k) ↑k) ↑(m.mod k)) ↑m
  -/
  rw [mul_comm]
  /-
    m k : PNat
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑k) (m.div k)) ↑(m.mod k)) ↑m
  -/
  exact div_add_mod _ _
  /-
    🎉 no goals
  -/


theorem mod_le (m k : ℕ+) : mod m k ≤ m ∧ mod m k ≤ k := by
  /-
    m k : PNat
    ⊢ And (LE.le (m.mod k) m) (LE.le (m.mod k) k)
  -/
  change (mod m k : ℕ) ≤ (m : ℕ) ∧ (mod m k : ℕ) ≤ (k : ℕ)
  /-
    m k : PNat
    ⊢ And (LE.le ↑(m.mod k) ↑m) (LE.le ↑(m.mod k) ↑k)
  -/
  rw [mod_coe]
  /-
    m k : PNat
    ⊢ And (LE.le (ite (Eq (HMod.hMod ↑m ↑k) 0) (↑k) (HMod.hMod ↑m ↑k)) ↑m) (LE.le  …
  -/
  split_ifs with h
    /-
      case pos
      m k : PNat
      h : Eq (HMod.hMod ↑m ↑k) 0
      ⊢ And (LE.le ↑k ↑m) (LE.le ↑k ↑k)
    -/
  · have hm : (m : ℕ) > 0 := m.pos
    /-
      case pos
      m k : PNat
      h : Eq (HMod.hMod ↑m ↑k) 0
      hm : GT.gt (↑m) 0
      ⊢ And (LE.le ↑k ↑m) (LE.le ↑k ↑k)
    -/
    rw [← Nat.mod_add_div (m : ℕ) (k : ℕ), h, zero_add] at hm ⊢
    /-
      case pos
      m k : PNat
      h : Eq (HMod.hMod ↑m ↑k) 0
      hm : GT.gt (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k)) 0
      ⊢ And (LE.le (↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) (LE.le ↑k ↑k)
    -/
    by_cases h₁ : (m : ℕ) / (k : ℕ) = 0
      /-
        case pos
        m k : PNat
        h : Eq (HMod.hMod ↑m ↑k) 0
        hm : GT.gt (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k)) 0
        h₁ : Eq (HDiv.hDiv ↑m ↑k) 0
        ⊢ And (LE.le (↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) (LE.le ↑k ↑k)
      -/
    · rw [h₁, mul_zero] at hm
      /-
        case pos
        m k : PNat
        h : Eq (HMod.hMod ↑m ↑k) 0
        hm : GT.gt 0 0
        h₁ : Eq (HDiv.hDiv ↑m ↑k) 0
        ⊢ And (LE.le (↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) (LE.le ↑k ↑k)
      -/
      exact (lt_irrefl _ hm).elim
      /-
        🎉 no goals
      -/
    · let h₂ : (k : ℕ) * 1 ≤ k * (m / k) :=
        -- Porting note: Specified type of `h₂` explicitly because `rw` could not unify
        -- `succ 0` with `1`.
        Nat.mul_le_mul_left (k : ℕ) (Nat.succ_le_of_lt (Nat.pos_of_ne_zero h₁))
      /-
        case neg
        m k : PNat
        h : Eq (HMod.hMod ↑m ↑k) 0
        hm : GT.gt (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k)) 0
        h₁ : Not (Eq (HDiv.hDiv ↑m ↑k) 0)
        h₂ : LE.le (HMul.hMul (↑k) 1) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k)) := Nat.mul_le …
        ⊢ And (LE.le (↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) (LE.le ↑k ↑k)
      -/
      rw [mul_one] at h₂
      /-
        case neg
        m k : PNat
        h : Eq (HMod.hMod ↑m ↑k) 0
        hm : GT.gt (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k)) 0
        h₁ : Not (Eq (HDiv.hDiv ↑m ↑k) 0)
        h₂ : LE.le (↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))
        ⊢ And (LE.le (↑k) (HMul.hMul (↑k) (HDiv.hDiv ↑m ↑k))) (LE.le ↑k ↑k)
      -/
      exact ⟨h₂, le_refl (k : ℕ)⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      m k : PNat
      h : Not (Eq (HMod.hMod ↑m ↑k) 0)
      ⊢ And (LE.le (HMod.hMod ↑m ↑k) ↑m) (LE.le (HMod.hMod ↑m ↑k) ↑k)
    -/
  · exact ⟨Nat.mod_le (m : ℕ) (k : ℕ), (Nat.mod_lt (m : ℕ) k.pos).le⟩
    /-
      🎉 no goals
    -/


theorem dvd_iff {k m : ℕ+} : k ∣ m ↔ (k : ℕ) ∣ (m : ℕ) := by
  /-
    k m : PNat
    ⊢ Iff (Dvd.dvd k m) (Dvd.dvd ↑k ↑m)
  -/
  constructor <;> intro h
    /-
      case mp
      k m : PNat
      h : Dvd.dvd k m
      ⊢ Dvd.dvd ↑k ↑m
    -/
  · rcases h with ⟨_, rfl⟩
    /-
      case mp.intro
      k w✝ : PNat
      ⊢ Dvd.dvd ↑k ↑(HMul.hMul k w✝)
    -/
    apply dvd_mul_right
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k m : PNat
      h : Dvd.dvd ↑k ↑m
      ⊢ Dvd.dvd k m
    -/
  · rcases h with ⟨a, h⟩
    obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (n := a) <| by
      rintro rfl
      simp only [mul_zero, ne_zero] at h
    /-
      case mpr.intro.intro
      k m : PNat
      n : Nat
      h : Eq (↑m) (HMul.hMul (↑k) n.succ)
      ⊢ Dvd.dvd k m
    -/
    use ⟨n.succ, n.succ_pos⟩
    /-
      case h
      k m : PNat
      n : Nat
      h : Eq (↑m) (HMul.hMul (↑k) n.succ)
      ⊢ Eq m (HMul.hMul k ⟨n.succ, ⋯⟩)
    -/
    rw [← coe_inj, h, mul_coe, mk_coe]
    /-
      🎉 no goals
    -/


theorem dvd_iff' {k m : ℕ+} : k ∣ m ↔ mod m k = k := by
  /-
    k m : PNat
    ⊢ Iff (Dvd.dvd k m) (Eq (m.mod k) k)
  -/
  rw [dvd_iff]
  /-
    k m : PNat
    ⊢ Iff (Dvd.dvd ↑k ↑m) (Eq (m.mod k) k)
  -/
  rw [Nat.dvd_iff_mod_eq_zero]; constructor
    /-
      case mp
      k m : PNat
      ⊢ Eq (HMod.hMod ↑m ↑k) 0 → Eq (m.mod k) k
    -/
  · intro h
    /-
      case mp
      k m : PNat
      h : Eq (HMod.hMod ↑m ↑k) 0
      ⊢ Eq (m.mod k) k
    -/
    apply PNat.eq
    /-
      case mp.a
      k m : PNat
      h : Eq (HMod.hMod ↑m ↑k) 0
      ⊢ Eq ↑(m.mod k) ↑k
    -/
    rw [mod_coe, if_pos h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k m : PNat
      ⊢ Eq (m.mod k) k → Eq (HMod.hMod ↑m ↑k) 0
    -/
  · intro h
    /-
      case mpr
      k m : PNat
      h : Eq (m.mod k) k
      ⊢ Eq (HMod.hMod ↑m ↑k) 0
    -/
    by_cases h' : (m : ℕ) % (k : ℕ) = 0
      /-
        case pos
        k m : PNat
        h : Eq (m.mod k) k
        h' : Eq (HMod.hMod ↑m ↑k) 0
        ⊢ Eq (HMod.hMod ↑m ↑k) 0
      -/
    · exact h'
      /-
        🎉 no goals
      -/
      /-
        case neg
        k m : PNat
        h : Eq (m.mod k) k
        h' : Not (Eq (HMod.hMod ↑m ↑k) 0)
        ⊢ Eq (HMod.hMod ↑m ↑k) 0
      -/
    · replace h : (mod m k : ℕ) = (k : ℕ) := congr_arg _ h
      /-
        case neg
        k m : PNat
        h' : Not (Eq (HMod.hMod ↑m ↑k) 0)
        h : Eq ↑(m.mod k) ↑k
        ⊢ Eq (HMod.hMod ↑m ↑k) 0
      -/
      rw [mod_coe, if_neg h'] at h
      /-
        case neg
        k m : PNat
        h' : Not (Eq (HMod.hMod ↑m ↑k) 0)
        h : Eq (HMod.hMod ↑m ↑k) ↑k
        ⊢ Eq (HMod.hMod ↑m ↑k) 0
      -/
      exact ((Nat.mod_lt (m : ℕ) k.pos).ne h).elim
      /-
        🎉 no goals
      -/


theorem le_of_dvd {m n : ℕ+} : m ∣ n → m ≤ n := by
  /-
    m n : PNat
    ⊢ Dvd.dvd m n → LE.le m n
  -/
  rw [dvd_iff']
  /-
    m n : PNat
    ⊢ Eq (n.mod m) m → LE.le m n
  -/
  intro h
  /-
    m n : PNat
    h : Eq (n.mod m) m
    ⊢ LE.le m n
  -/
  rw [← h]
  /-
    m n : PNat
    h : Eq (n.mod m) m
    ⊢ LE.le (n.mod m) n
  -/
  apply (mod_le n m).left
  /-
    🎉 no goals
  -/


theorem mul_div_exact {m k : ℕ+} (h : k ∣ m) : k * divExact m k = m := by
  /-
    m k : PNat
    h : Dvd.dvd k m
    ⊢ Eq (HMul.hMul k (m.divExact k)) m
  -/
  apply PNat.eq; rw [mul_coe]
  /-
    case a
    m k : PNat
    h : Dvd.dvd k m
    ⊢ Eq (HMul.hMul ↑k ↑(m.divExact k)) ↑m
  -/
  change (k : ℕ) * (div m k).succ = m
  /-
    case a
    m k : PNat
    h : Dvd.dvd k m
    ⊢ Eq (HMul.hMul (↑k) (m.div k).succ) ↑m
  -/
  rw [← div_add_mod m k, dvd_iff'.mp h, Nat.mul_succ]
  /-
    🎉 no goals
  -/


theorem dvd_antisymm {m n : ℕ+} : m ∣ n → n ∣ m → m = n := fun hmn hnm =>
  (le_of_dvd hmn).antisymm (le_of_dvd hnm)


theorem dvd_one_iff (n : ℕ+) : n ∣ 1 ↔ n = 1 :=
  ⟨fun h => dvd_antisymm h (one_dvd n), fun h => h.symm ▸ dvd_refl 1⟩


theorem pos_of_div_pos {n : ℕ+} {a : ℕ} (h : a ∣ n) : 0 < a := by
  /-
    n : PNat
    a : Nat
    h : Dvd.dvd a ↑n
    ⊢ LT.lt 0 a
  -/
  apply pos_iff_ne_zero.2
  /-
    n : PNat
    a : Nat
    h : Dvd.dvd a ↑n
    ⊢ Ne a 0
  -/
  intro hzero
  /-
    n : PNat
    a : Nat
    h : Dvd.dvd a ↑n
    hzero : Eq a 0
    ⊢ False
  -/
  rw [hzero] at h
  /-
    n : PNat
    a : Nat
    h : Dvd.dvd 0 ↑n
    hzero : Eq a 0
    ⊢ False
  -/
  exact PNat.ne_zero n (eq_zero_of_zero_dvd h)
  /-
    🎉 no goals
  -/


