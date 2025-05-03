/-- A type synonym for ordinals with natural addition and multiplication. -/
def NatOrdinal : Type _ :=
  -- Porting note: used to derive LinearOrder & SuccOrder but need to manually define
  Ordinal deriving Zero, Inhabited, One, WellFoundedRelation


instance NatOrdinal.instLinearOrder : LinearOrder NatOrdinal := Ordinal.instLinearOrder

instance NatOrdinal.instSuccOrder : SuccOrder NatOrdinal := Ordinal.instSuccOrder

instance NatOrdinal.instOrderBot : OrderBot NatOrdinal := Ordinal.instOrderBot

instance NatOrdinal.instNoMaxOrder : NoMaxOrder NatOrdinal := Ordinal.instNoMaxOrder

instance NatOrdinal.instZeroLEOneClass : ZeroLEOneClass NatOrdinal := Ordinal.instZeroLEOneClass

instance NatOrdinal.instNeZeroOne : NeZero (1 : NatOrdinal) := Ordinal.instNeZeroOne


/-- The identity function between `Ordinal` and `NatOrdinal`. -/
@[match_pattern]
def Ordinal.toNatOrdinal : Ordinal ≃o NatOrdinal :=
  OrderIso.refl _


/-- The identity function between `NatOrdinal` and `Ordinal`. -/
@[match_pattern]
def NatOrdinal.toOrdinal : NatOrdinal ≃o Ordinal :=
  OrderIso.refl _


@[simp]
theorem toOrdinal_symm_eq : NatOrdinal.toOrdinal.symm = Ordinal.toNatOrdinal :=
  rfl


@[simp]
theorem toOrdinal_toNatOrdinal (a : NatOrdinal) : a.toOrdinal.toNatOrdinal = a :=
  rfl


theorem lt_wf : @WellFounded NatOrdinal (· < ·) :=
  Ordinal.lt_wf


instance : WellFoundedLT NatOrdinal :=
  Ordinal.wellFoundedLT


instance : ConditionallyCompleteLinearOrderBot NatOrdinal :=
  WellFoundedLT.conditionallyCompleteLinearOrderBot _


@[simp]
theorem bot_eq_zero : ⊥ = 0 :=
  rfl


@[simp]
theorem toOrdinal_zero : toOrdinal 0 = 0 :=
  rfl


@[simp]
theorem toOrdinal_one : toOrdinal 1 = 1 :=
  rfl


@[simp]
theorem toOrdinal_eq_zero {a} : toOrdinal a = 0 ↔ a = 0 :=
  Iff.rfl


@[simp]
theorem toOrdinal_eq_one {a} : toOrdinal a = 1 ↔ a = 1 :=
  Iff.rfl


@[simp]
theorem toOrdinal_max (a b : NatOrdinal) : toOrdinal (max a b) = max (toOrdinal a) (toOrdinal b) :=
  rfl


@[simp]
theorem toOrdinal_min (a b : NatOrdinal) : toOrdinal (min a b) = min (toOrdinal a) (toOrdinal b) :=
  rfl


theorem succ_def (a : NatOrdinal) : succ a = toNatOrdinal (toOrdinal a + 1) :=
  rfl


/-- A recursor for `NatOrdinal`. Use as `induction x`. -/
@[elab_as_elim, cases_eliminator, induction_eliminator]
protected def rec {β : NatOrdinal → Sort*} (h : ∀ a, β (toNatOrdinal a)) : ∀ a, β a := fun a =>
  h (toOrdinal a)


/-- `Ordinal.induction` but for `NatOrdinal`. -/
theorem induction {p : NatOrdinal → Prop} : ∀ (i) (_ : ∀ j, (∀ k, k < j → p k) → p j), p i :=
  Ordinal.induction


@[simp]
theorem toNatOrdinal_symm_eq : toNatOrdinal.symm = NatOrdinal.toOrdinal :=
  rfl


@[simp]
theorem toNatOrdinal_toOrdinal (a : Ordinal) : a.toNatOrdinal.toOrdinal = a :=
  rfl


@[simp]
theorem toNatOrdinal_zero : toNatOrdinal 0 = 0 :=
  rfl


@[simp]
theorem toNatOrdinal_one : toNatOrdinal 1 = 1 :=
  rfl


@[simp]
theorem toNatOrdinal_eq_zero (a) : toNatOrdinal a = 0 ↔ a = 0 :=
  Iff.rfl


@[simp]
theorem toNatOrdinal_eq_one (a) : toNatOrdinal a = 1 ↔ a = 1 :=
  Iff.rfl


@[simp]
theorem toNatOrdinal_max (a b : Ordinal) :
    toNatOrdinal (max a b) = max (toNatOrdinal a) (toNatOrdinal b) :=
  rfl


@[simp]
theorem toNatOrdinal_min (a b : Ordinal) :
    toNatOrdinal (min a b) = min (toNatOrdinal a) (toNatOrdinal b) :=
  rfl


/-- Natural addition on ordinals `a ♯ b`, also known as the Hessenberg sum, is recursively defined
as the least ordinal greater than `a' ♯ b` and `a ♯ b'` for all `a' < a` and `b' < b`. In contrast
to normal ordinal addition, it is commutative.

Natural addition can equivalently be characterized as the ordinal resulting from adding up
corresponding coefficients in the Cantor normal forms of `a` and `b`. -/
noncomputable def nadd (a b : Ordinal) : Ordinal :=
  max (blsub.{u, u} a fun a' _ => nadd a' b) (blsub.{u, u} b fun b' _ => nadd a b')
termination_by (a, b)


@[inherit_doc]
scoped[NaturalOps] infixl:65 " ♯ " => Ordinal.nadd


/-- Natural multiplication on ordinals `a ⨳ b`, also known as the Hessenberg product, is recursively
defined as the least ordinal such that `a ⨳ b ♯ a' ⨳ b'` is greater than `a' ⨳ b ♯ a ⨳ b'` for all
`a' < a` and `b < b'`. In contrast to normal ordinal multiplication, it is commutative and
distributive (over natural addition).

Natural multiplication can equivalently be characterized as the ordinal resulting from multiplying
the Cantor normal forms of `a` and `b` as if they were polynomials in `ω`. Addition of exponents is
done via natural addition. -/
noncomputable def nmul (a b : Ordinal.{u}) : Ordinal.{u} :=
  sInf {c | ∀ a' < a, ∀ b' < b, nmul a' b ♯ nmul a b' < c ♯ nmul a' b'}
termination_by (a, b)


@[inherit_doc]
scoped[NaturalOps] infixl:70 " ⨳ " => Ordinal.nmul


theorem nadd_def (a b : Ordinal) :
    a ♯ b = max (blsub.{u, u} a fun a' _ => a' ♯ b) (blsub.{u, u} b fun b' _ => a ♯ b') := by
  /-
    a b : Ordinal.{u}
    ⊢ Eq (a.nadd b) (Max.max (a.blsub fun a' x => a'.nadd b) (b.blsub fun b' x =>  …
  -/
  rw [nadd]
  /-
    🎉 no goals
  -/


theorem lt_nadd_iff : a < b ♯ c ↔ (∃ b' < b, a ≤ b' ♯ c) ∨ ∃ c' < c, a ≤ b ♯ c' := by
  /-
    a b c : Ordinal.{u}
    ⊢ Iff (LT.lt a (b.nadd c)) (Or (Exists fun b' => And (LT.lt b' b) (LE.le a (b' …
  -/
  rw [nadd_def]
  /-
    a b c : Ordinal.{u}
    ⊢ Iff (LT.lt a (Max.max (b.blsub fun a' x => a'.nadd c) (c.blsub fun b' x => b …
  -/
  simp [lt_blsub_iff]
  /-
    🎉 no goals
  -/


theorem nadd_le_iff : b ♯ c ≤ a ↔ (∀ b' < b, b' ♯ c < a) ∧ ∀ c' < c, b ♯ c' < a := by
  /-
    a b c : Ordinal.{u}
    ⊢ Iff (LE.le (b.nadd c) a) (And (∀ (b' : Ordinal.{u}), LT.lt b' b → LT.lt (b'. …
  -/
  rw [nadd_def]
  /-
    a b c : Ordinal.{u}
    ⊢ Iff (LE.le (Max.max (b.blsub fun a' x => a'.nadd c) (c.blsub fun b' x => b.n …
  -/
  simp [blsub_le_iff]
  /-
    🎉 no goals
  -/


theorem nadd_lt_nadd_left (h : b < c) (a) : a ♯ b < a ♯ c :=
  lt_nadd_iff.2 (Or.inr ⟨b, h, le_rfl⟩)


theorem nadd_lt_nadd_right (h : b < c) (a) : b ♯ a < c ♯ a :=
  lt_nadd_iff.2 (Or.inl ⟨b, h, le_rfl⟩)


theorem nadd_le_nadd_left (h : b ≤ c) (a) : a ♯ b ≤ a ♯ c := by
  /-
    b c : Ordinal.{u}
    h : LE.le b c
    a : Ordinal.{u}
    ⊢ LE.le (a.nadd b) (a.nadd c)
  -/
  rcases lt_or_eq_of_le h with (h | rfl)
    /-
      case inl
      b c : Ordinal.{u}
      h✝ : LE.le b c
      a : Ordinal.{u}
      h : LT.lt b c
      ⊢ LE.le (a.nadd b) (a.nadd c)
    -/
  · exact (nadd_lt_nadd_left h a).le
    /-
      🎉 no goals
    -/
    /-
      case inr
      b a : Ordinal.{u}
      h : LE.le b b
      ⊢ LE.le (a.nadd b) (a.nadd b)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


theorem nadd_le_nadd_right (h : b ≤ c) (a) : b ♯ a ≤ c ♯ a := by
  /-
    b c : Ordinal.{u}
    h : LE.le b c
    a : Ordinal.{u}
    ⊢ LE.le (b.nadd a) (c.nadd a)
  -/
  rcases lt_or_eq_of_le h with (h | rfl)
    /-
      case inl
      b c : Ordinal.{u}
      h✝ : LE.le b c
      a : Ordinal.{u}
      h : LT.lt b c
      ⊢ LE.le (b.nadd a) (c.nadd a)
    -/
  · exact (nadd_lt_nadd_right h a).le
    /-
      🎉 no goals
    -/
    /-
      case inr
      b a : Ordinal.{u}
      h : LE.le b b
      ⊢ LE.le (b.nadd a) (b.nadd a)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


theorem nadd_comm (a b) : a ♯ b = b ♯ a := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Eq (a.nadd b) (b.nadd a)
  -/
  rw [nadd_def, nadd_def, max_comm]
  /-
    a b : Ordinal.{u_1}
    ⊢ Eq (Max.max (b.blsub fun b' x => a.nadd b') (a.blsub fun a' x => a'.nadd b)) …
  -/
                    /-
                      🎉 no goals
                    -/
  congr <;> ext <;> apply nadd_comm
                    /-
                      🎉 no goals
                    -/
termination_by (a, b)


theorem blsub_nadd_of_mono {f : ∀ c < a ♯ b, Ordinal.{max u v}}
    (hf : ∀ {i j} (hi hj), i ≤ j → f i hi ≤ f j hj) :
    -- Porting note: needed to add universe hint blsub.{u,v} in the line below
    blsub.{u,v} _ f =
      max (blsub.{u, v} a fun a' ha' => f (a' ♯ b) <| nadd_lt_nadd_right ha' b)
        (blsub.{u, v} b fun b' hb' => f (a ♯ b') <| nadd_lt_nadd_left hb' a) := by
  /-
    a b : Ordinal.{u}
    f : (c : Ordinal.{u}) → LT.lt c (a.nadd b) → Ordinal.{max u v}
    hf : ∀ {i j : Ordinal.{u}} (hi : LT.lt i (a.nadd b)) (hj : LT.lt j (a.nadd b)) …
    ⊢ Eq ((a.nadd b).blsub f) (Max.max (a.blsub fun a' ha' => f (a'.nadd b) ⋯) (b. …
  -/
  apply (blsub_le_iff.2 fun i h => _).antisymm (max_le _ _)
    /-
      a b : Ordinal.{u}
      f : (c : Ordinal.{u}) → LT.lt c (a.nadd b) → Ordinal.{max u v}
      hf : ∀ {i j : Ordinal.{u}} (hi : LT.lt i (a.nadd b)) (hj : LT.lt j (a.nadd b)) …
      ⊢ ∀ (i : Ordinal.{u}) (h : LT.lt i (a.nadd b)), LT.lt (f i h) (Max.max (a.blsu …
    -/
  · intro i h
    /-
      a b : Ordinal.{u}
      f : (c : Ordinal.{u}) → LT.lt c (a.nadd b) → Ordinal.{max u v}
      hf : ∀ {i j : Ordinal.{u}} (hi : LT.lt i (a.nadd b)) (hj : LT.lt j (a.nadd b)) …
      i : Ordinal.{u}
      h : LT.lt i (a.nadd b)
      ⊢ LT.lt (f i h) (Max.max (a.blsub fun a' ha' => f (a'.nadd b) ⋯) (b.blsub fun  …
    -/
    rcases lt_nadd_iff.1 h with (⟨a', ha', hi⟩ | ⟨b', hb', hi⟩)
      /-
        case inl.intro.intro
        a b : Ordinal.{u}
        f : (c : Ordinal.{u}) → LT.lt c (a.nadd b) → Ordinal.{max u v}
        hf : ∀ {i j : Ordinal.{u}} (hi : LT.lt i (a.nadd b)) (hj : LT.lt j (a.nadd b)) …
        i : Ordinal.{u}
        h : LT.lt i (a.nadd b)
        a' : Ordinal.{u}
        ha' : LT.lt a' a
        hi : LE.le i (a'.nadd b)
        ⊢ LT.lt (f i h) (Max.max (a.blsub fun a' ha' => f (a'.nadd b) ⋯) (b.blsub fun  …
      -/
    · exact lt_max_of_lt_left ((hf h (nadd_lt_nadd_right ha' b) hi).trans_lt (lt_blsub _ _ ha'))
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro
        a b : Ordinal.{u}
        f : (c : Ordinal.{u}) → LT.lt c (a.nadd b) → Ordinal.{max u v}
        hf : ∀ {i j : Ordinal.{u}} (hi : LT.lt i (a.nadd b)) (hj : LT.lt j (a.nadd b)) …
        i : Ordinal.{u}
        h : LT.lt i (a.nadd b)
        b' : Ordinal.{u}
        hb' : LT.lt b' b
        hi : LE.le i (a.nadd b')
        ⊢ LT.lt (f i h) (Max.max (a.blsub fun a' ha' => f (a'.nadd b) ⋯) (b.blsub fun  …
      -/
    · exact lt_max_of_lt_right ((hf h (nadd_lt_nadd_left hb' a) hi).trans_lt (lt_blsub _ _ hb'))
      /-
        🎉 no goals
      -/
  all_goals
    apply blsub_le_of_brange_subset.{u, u, v}
    rintro c ⟨d, hd, rfl⟩
    apply mem_brange_self


theorem nadd_assoc (a b c) : a ♯ b ♯ c = a ♯ (b ♯ c) := by
  /-
    a b c : Ordinal.{u_1}
    ⊢ Eq ((a.nadd b).nadd c) (a.nadd (b.nadd c))
  -/
  rw [nadd_def a (b ♯ c), nadd_def, blsub_nadd_of_mono, blsub_nadd_of_mono, max_assoc]
    /-
      a b c : Ordinal.{u_1}
      ⊢ Eq (Max.max (a.blsub fun a' ha' => (a'.nadd b).nadd c) (Max.max (b.blsub fun …
    -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
  · congr <;> ext <;> apply nadd_assoc
                      /-
                        🎉 no goals
                      -/
    /-
      case hf
      a b c : Ordinal.{u_1}
      ⊢ ∀ {i j : Ordinal.{u_1}}, LT.lt i (b.nadd c) → LT.lt j (b.nadd c) → LE.le i j …
    -/
  · exact fun _ _ h => nadd_le_nadd_left h a
    /-
      🎉 no goals
    -/
    /-
      case hf
      a b c : Ordinal.{u_1}
      ⊢ ∀ {i j : Ordinal.{u_1}}, LT.lt i (a.nadd b) → LT.lt j (a.nadd b) → LE.le i j …
    -/
  · exact fun _ _ h => nadd_le_nadd_right h c
    /-
      🎉 no goals
    -/
termination_by (a, b, c)


@[simp]
theorem nadd_zero : a ♯ 0 = a := by
  /-
    a : Ordinal.{u}
    ⊢ Eq (a.nadd 0) a
  -/
  induction' a using Ordinal.induction with a IH
  /-
    case h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 0) k
    ⊢ Eq (a.nadd 0) a
  -/
  rw [nadd_def, blsub_zero, max_zero_right]
  /-
    case h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 0) k
    ⊢ Eq (a.blsub fun a' x => a'.nadd 0) a
  -/
  convert blsub_id a
  /-
    case h.e'_2.h.e'_2.h.h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 0) k
    x✝¹ : Ordinal.{u}
    x✝ : LT.lt x✝¹ a
    ⊢ Eq (x✝¹.nadd 0) x✝¹
  -/
  rename_i hb
  /-
    case h.e'_2.h.e'_2.h.h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 0) k
    x✝ : Ordinal.{u}
    hb : LT.lt x✝ a
    ⊢ Eq (x✝.nadd 0) x✝
  -/
  exact IH _ hb
  /-
    🎉 no goals
  -/


@[simp]
                                    /-
                                      a : Ordinal.{u}
                                      ⊢ Eq (Ordinal.nadd 0 a) a
                                    -/
theorem zero_nadd : 0 ♯ a = a := by rw [nadd_comm, nadd_zero]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem nadd_one : a ♯ 1 = succ a := by
  /-
    a : Ordinal.{u}
    ⊢ Eq (a.nadd 1) (Order.succ a)
  -/
  induction' a using Ordinal.induction with a IH
  /-
    case h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 1) (Order.succ k)
    ⊢ Eq (a.nadd 1) (Order.succ a)
  -/
  rw [nadd_def, blsub_one, nadd_zero, max_eq_right_iff, blsub_le_iff]
  /-
    case h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 1) (Order.succ k)
    ⊢ ∀ (i : Ordinal.{u}), LT.lt i a → LT.lt (i.nadd 1) (Order.succ a)
  -/
  intro i hi
  /-
    case h
    a✝ a : Ordinal.{u}
    IH : ∀ (k : Ordinal.{u}), LT.lt k a → Eq (k.nadd 1) (Order.succ k)
    i : Ordinal.{u}
    hi : LT.lt i a
    ⊢ LT.lt (i.nadd 1) (Order.succ a)
  -/
  rwa [IH i hi, succ_lt_succ_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                        /-
                                          a : Ordinal.{u}
                                          ⊢ Eq (Ordinal.nadd 1 a) (Order.succ a)
                                        -/
theorem one_nadd : 1 ♯ a = succ a := by rw [nadd_comm, nadd_one]
                                        /-
                                          🎉 no goals
                                        -/


                                                    /-
                                                      a b : Ordinal.{u}
                                                      ⊢ Eq (a.nadd (Order.succ b)) (Order.succ (a.nadd b))
                                                    -/
theorem nadd_succ : a ♯ succ b = succ (a ♯ b) := by rw [← nadd_one (a ♯ b), nadd_assoc, nadd_one]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                    /-
                                                      a b : Ordinal.{u}
                                                      ⊢ Eq ((Order.succ a).nadd b) (Order.succ (a.nadd b))
                                                    -/
theorem succ_nadd : succ a ♯ b = succ (a ♯ b) := by rw [← one_nadd (a ♯ b), ← nadd_assoc, one_nadd]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem nadd_nat (n : ℕ) : a ♯ n = a + n := by
  /-
    a : Ordinal.{u}
    n : Nat
    ⊢ Eq (a.nadd ↑n) (HAdd.hAdd a ↑n)
  -/
  induction' n with n hn
    /-
      case zero
      a : Ordinal.{u}
      ⊢ Eq (a.nadd ↑0) (HAdd.hAdd a ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      a : Ordinal.{u}
      n : Nat
      hn : Eq (a.nadd ↑n) (HAdd.hAdd a ↑n)
      ⊢ Eq (a.nadd ↑(HAdd.hAdd n 1)) (HAdd.hAdd a ↑(HAdd.hAdd n 1))
    -/
  · rw [Nat.cast_succ, add_one_eq_succ, nadd_succ, add_succ, hn]
    /-
      🎉 no goals
    -/


@[simp]
                                                /-
                                                  a : Ordinal.{u}
                                                  n : Nat
                                                  ⊢ Eq ((↑n).nadd a) (HAdd.hAdd a ↑n)
                                                -/
theorem nat_nadd (n : ℕ) : ↑n ♯ a = a + n := by rw [nadd_comm, nadd_nat]
                                                /-
                                                  🎉 no goals
                                                -/


theorem add_le_nadd : a + b ≤ a ♯ b := by
  induction b using limitRecOn with
  | H₁ => simp
  | H₂ c h =>
    rwa [add_succ, nadd_succ, succ_le_succ_iff]
  | H₃ c hc H =>
    simp_rw [← IsNormal.blsub_eq.{u, u} (isNormal_add_right a) hc, blsub_le_iff]
    exact fun i hi => (H i hi).trans_lt (nadd_lt_nadd_left hi a)


instance : Add NatOrdinal := ⟨nadd⟩

instance : SuccAddOrder NatOrdinal := ⟨fun x => (nadd_one x).symm⟩


instance : AddLeftStrictMono NatOrdinal.{u} :=
  ⟨fun a _ _ h => nadd_lt_nadd_left h a⟩


instance : AddLeftMono NatOrdinal.{u} :=
  ⟨fun a _ _ h => nadd_le_nadd_left h a⟩


instance : AddLeftReflectLE NatOrdinal.{u} :=
  ⟨fun a b c h => by
    /-
      a b c : NatOrdinal
      h : LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
      ⊢ LE.le b c
    -/
    by_contra! h'
    /-
      a b c : NatOrdinal
      h : LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
      h' : LT.lt c b
      ⊢ False
    -/
    exact h.not_lt (add_lt_add_left h' a)⟩
    /-
      🎉 no goals
    -/


instance : OrderedCancelAddCommMonoid NatOrdinal :=
  { NatOrdinal.instLinearOrder with
    add := (· + ·)
    add_assoc := nadd_assoc
    add_le_add_left := fun _ _ => add_le_add_left
    le_of_add_le_add_left := fun _ _ _ => le_of_add_le_add_left
    zero := 0
    zero_add := zero_nadd
    add_zero := nadd_zero
    add_comm := nadd_comm
    nsmul := nsmulRec }


instance : AddMonoidWithOne NatOrdinal :=
  AddMonoidWithOne.unary


@[deprecated Order.succ_eq_add_one (since := "2024-09-04")]
theorem add_one_eq_succ (a : NatOrdinal) : a + 1 = succ a :=
  (Order.succ_eq_add_one a).symm


@[simp]
theorem toOrdinal_cast_nat (n : ℕ) : toOrdinal n = n := by
  /-
    n : Nat
    ⊢ Eq (NatOrdinal.toOrdinal ↑n) ↑n
  -/
  induction' n with n hn
    /-
      case zero
      ⊢ Eq (NatOrdinal.toOrdinal ↑0) ↑0
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      hn : Eq (NatOrdinal.toOrdinal ↑n) ↑n
      ⊢ Eq (NatOrdinal.toOrdinal ↑(HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)
    -/
  · change (toOrdinal n) ♯ 1 = n + 1
    /-
      case succ
      n : Nat
      hn : Eq (NatOrdinal.toOrdinal ↑n) ↑n
      ⊢ Eq ((NatOrdinal.toOrdinal ↑n).nadd 1) (HAdd.hAdd (↑n) 1)
    -/
    rw [hn]; exact nadd_one n
             /-
               🎉 no goals
             -/


theorem nadd_eq_add (a b : Ordinal) : a ♯ b = toOrdinal (toNatOrdinal a + toNatOrdinal b) :=
  rfl


@[simp]
theorem toNatOrdinal_cast_nat (n : ℕ) : toNatOrdinal n = n := by
  /-
    n : Nat
    ⊢ Eq (Ordinal.toNatOrdinal ↑n) ↑n
  -/
  rw [← toOrdinal_cast_nat n]
  /-
    n : Nat
    ⊢ Eq (Ordinal.toNatOrdinal (NatOrdinal.toOrdinal ↑n)) ↑n
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem lt_of_nadd_lt_nadd_left : ∀ {a b c}, a ♯ b < a ♯ c → b < c :=
  @lt_of_add_lt_add_left NatOrdinal _ _ _


theorem lt_of_nadd_lt_nadd_right : ∀ {a b c}, b ♯ a < c ♯ a → b < c :=
  @lt_of_add_lt_add_right NatOrdinal _ _ _


theorem le_of_nadd_le_nadd_left : ∀ {a b c}, a ♯ b ≤ a ♯ c → b ≤ c :=
  @le_of_add_le_add_left NatOrdinal _ _ _


theorem le_of_nadd_le_nadd_right : ∀ {a b c}, b ♯ a ≤ c ♯ a → b ≤ c :=
  @le_of_add_le_add_right NatOrdinal _ _ _


theorem nadd_lt_nadd_iff_left : ∀ (a) {b c}, a ♯ b < a ♯ c ↔ b < c :=
  @add_lt_add_iff_left NatOrdinal _ _ _ _


theorem nadd_lt_nadd_iff_right : ∀ (a) {b c}, b ♯ a < c ♯ a ↔ b < c :=
  @add_lt_add_iff_right NatOrdinal _ _ _ _


theorem nadd_le_nadd_iff_left : ∀ (a) {b c}, a ♯ b ≤ a ♯ c ↔ b ≤ c :=
  @add_le_add_iff_left NatOrdinal _ _ _ _


theorem nadd_le_nadd_iff_right : ∀ (a) {b c}, b ♯ a ≤ c ♯ a ↔ b ≤ c :=
  @_root_.add_le_add_iff_right NatOrdinal _ _ _ _


theorem nadd_le_nadd : ∀ {a b c d}, a ≤ b → c ≤ d → a ♯ c ≤ b ♯ d :=
  @add_le_add NatOrdinal _ _ _ _


theorem nadd_lt_nadd : ∀ {a b c d}, a < b → c < d → a ♯ c < b ♯ d :=
  @add_lt_add NatOrdinal _ _ _ _


theorem nadd_lt_nadd_of_lt_of_le : ∀ {a b c d}, a < b → c ≤ d → a ♯ c < b ♯ d :=
  @add_lt_add_of_lt_of_le NatOrdinal _ _ _ _


theorem nadd_lt_nadd_of_le_of_lt : ∀ {a b c d}, a ≤ b → c < d → a ♯ c < b ♯ d :=
  @add_lt_add_of_le_of_lt NatOrdinal _ _ _ _


theorem nadd_left_cancel : ∀ {a b c}, a ♯ b = a ♯ c → b = c :=
  @_root_.add_left_cancel NatOrdinal _ _


theorem nadd_right_cancel : ∀ {a b c}, a ♯ b = c ♯ b → a = c :=
  @_root_.add_right_cancel NatOrdinal _ _


theorem nadd_left_cancel_iff : ∀ {a b c}, a ♯ b = a ♯ c ↔ b = c :=
  @add_left_cancel_iff NatOrdinal _ _


theorem nadd_right_cancel_iff : ∀ {a b c}, b ♯ a = c ♯ a ↔ b = c :=
  @add_right_cancel_iff NatOrdinal _ _


                                             /-
                                               a b : Ordinal.{u_1}
                                               ⊢ LE.le a (b.nadd a)
                                             -/
theorem le_nadd_self {a b} : a ≤ b ♯ a := by simpa using nadd_le_nadd_right (Ordinal.zero_le b) a
                                             /-
                                               🎉 no goals
                                             -/


theorem le_nadd_left {a b c} (h : a ≤ c) : a ≤ b ♯ c :=
  le_nadd_self.trans (nadd_le_nadd_left h b)


                                             /-
                                               a b : Ordinal.{u_1}
                                               ⊢ LE.le a (a.nadd b)
                                             -/
theorem le_self_nadd {a b} : a ≤ a ♯ b := by simpa using nadd_le_nadd_left (Ordinal.zero_le b) a
                                             /-
                                               🎉 no goals
                                             -/


theorem le_nadd_right {a b c} (h : a ≤ b) : a ≤ b ♯ c :=
  le_self_nadd.trans (nadd_le_nadd_right h c)


theorem nadd_left_comm : ∀ a b c, a ♯ (b ♯ c) = b ♯ (a ♯ c) :=
  @add_left_comm NatOrdinal _


theorem nadd_right_comm : ∀ a b c, a ♯ b ♯ c = a ♯ c ♯ b :=
  @add_right_comm NatOrdinal _


@[deprecated "avoid using the definition of `nmul` directly" (since := "2024-11-19")]
theorem nmul_def (a b : Ordinal) :
    a ⨳ b = sInf {c | ∀ a' < a, ∀ b' < b, a' ⨳ b ♯ a ⨳ b' < c ♯ a' ⨳ b'} := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Eq (a.nmul b) (InfSet.sInf (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a'  …
  -/
  rw [nmul]
  /-
    🎉 no goals
  -/


/-- The set in the definition of `nmul` is nonempty. -/
private theorem nmul_nonempty (a b : Ordinal.{u}) :
    {c : Ordinal.{u} | ∀ a' < a, ∀ b' < b, a' ⨳ b ♯ a ⨳ b' < c ♯ a' ⨳ b'}.Nonempty := by
  obtain ⟨c, hc⟩ : BddAbove ((fun x ↦ x.1 ⨳ b ♯ a ⨳ x.2) '' Set.Iio a ×ˢ Set.Iio b) :=
    bddAbove_of_small _
  exact ⟨_, fun x hx y hy ↦
    (lt_succ_of_le <| hc <| Set.mem_image_of_mem _ <| Set.mk_mem_prod hx hy).trans_le le_self_nadd⟩


theorem nmul_nadd_lt {a' b' : Ordinal} (ha : a' < a) (hb : b' < b) :
    a' ⨳ b ♯ a ⨳ b' < a ⨳ b ♯ a' ⨳ b' := by
  /-
    a b a' b' : Ordinal.{u}
    ha : LT.lt a' a
    hb : LT.lt b' b
    ⊢ LT.lt ((a'.nmul b).nadd (a.nmul b')) ((a.nmul b).nadd (a'.nmul b'))
  -/
  conv_rhs => rw [nmul]
  /-
    a b a' b' : Ordinal.{u}
    ha : LT.lt a' a
    hb : LT.lt b' b
    ⊢ LT.lt ((a'.nmul b).nadd (a.nmul b')) ((InfSet.sInf (setOf fun c => ∀ (a' : O …
  -/
  exact csInf_mem (nmul_nonempty a b) a' ha b' hb
  /-
    🎉 no goals
  -/


theorem nmul_nadd_le {a' b' : Ordinal} (ha : a' ≤ a) (hb : b' ≤ b) :
    a' ⨳ b ♯ a ⨳ b' ≤ a ⨳ b ♯ a' ⨳ b' := by
  /-
    a b a' b' : Ordinal.{u}
    ha : LE.le a' a
    hb : LE.le b' b
    ⊢ LE.le ((a'.nmul b).nadd (a.nmul b')) ((a.nmul b).nadd (a'.nmul b'))
  -/
  rcases lt_or_eq_of_le ha with (ha | rfl)
    /-
      case inl
      a b a' b' : Ordinal.{u}
      ha✝ : LE.le a' a
      hb : LE.le b' b
      ha : LT.lt a' a
      ⊢ LE.le ((a'.nmul b).nadd (a.nmul b')) ((a.nmul b).nadd (a'.nmul b'))
    -/
  · rcases lt_or_eq_of_le hb with (hb | rfl)
      /-
        case inl.inl
        a b a' b' : Ordinal.{u}
        ha✝ : LE.le a' a
        hb✝ : LE.le b' b
        ha : LT.lt a' a
        hb : LT.lt b' b
        ⊢ LE.le ((a'.nmul b).nadd (a.nmul b')) ((a.nmul b).nadd (a'.nmul b'))
      -/
    · exact (nmul_nadd_lt ha hb).le
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        a a' b' : Ordinal.{u}
        ha✝ : LE.le a' a
        ha : LT.lt a' a
        hb : LE.le b' b'
        ⊢ LE.le ((a'.nmul b').nadd (a.nmul b')) ((a.nmul b').nadd (a'.nmul b'))
      -/
    · rw [nadd_comm]
      /-
        🎉 no goals
      -/
    /-
      case inr
      b a' b' : Ordinal.{u}
      hb : LE.le b' b
      ha : LE.le a' a'
      ⊢ LE.le ((a'.nmul b).nadd (a'.nmul b')) ((a'.nmul b).nadd (a'.nmul b'))
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/


theorem lt_nmul_iff : c < a ⨳ b ↔ ∃ a' < a, ∃ b' < b, c ♯ a' ⨳ b' ≤ a' ⨳ b ♯ a ⨳ b' := by
  /-
    a b c : Ordinal.{u}
    ⊢ Iff (LT.lt c (a.nmul b)) (Exists fun a' => And (LT.lt a' a) (Exists fun b' = …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      a b c : Ordinal.{u}
      h : LT.lt c (a.nmul b)
      ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (LE.le  …
    -/
  · rw [nmul] at h
    /-
      case refine_1
      a b c : Ordinal.{u}
      h : LT.lt c (InfSet.sInf (setOf fun c => ∀ (a' : Ordinal.{u}), LT.lt a' a → ∀  …
      ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (LE.le  …
    -/
    simpa using not_mem_of_lt_csInf h ⟨0, fun _ _ => bot_le⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b c : Ordinal.{u}
      ⊢ (Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (LE.le …
    -/
  · rintro ⟨a', ha, b', hb, h⟩
    /-
      case refine_2.intro.intro.intro.intro
      a b c a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      h : LE.le (c.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      ⊢ LT.lt c (a.nmul b)
    -/
    have := h.trans_lt (nmul_nadd_lt ha hb)
    /-
      case refine_2.intro.intro.intro.intro
      a b c a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      h : LE.le (c.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      this : LT.lt (c.nadd (a'.nmul b')) ((a.nmul b).nadd (a'.nmul b'))
      ⊢ LT.lt c (a.nmul b)
    -/
    rwa [nadd_lt_nadd_iff_right] at this
    /-
      🎉 no goals
    -/


theorem nmul_le_iff : a ⨳ b ≤ c ↔ ∀ a' < a, ∀ b' < b, a' ⨳ b ♯ a ⨳ b' < c ♯ a' ⨳ b' := by
  /-
    a b c : Ordinal.{u}
    ⊢ Iff (LE.le (a.nmul b) c) (∀ (a' : Ordinal.{u}), LT.lt a' a → ∀ (b' : Ordinal …
  -/
  rw [← not_iff_not]; simp [lt_nmul_iff]
                      /-
                        🎉 no goals
                      -/


theorem nmul_comm (a b) : a ⨳ b = b ⨳ a := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Eq (a.nmul b) (b.nmul a)
  -/
  rw [nmul, nmul]
  /-
    a b : Ordinal.{u_1}
    ⊢ Eq (InfSet.sInf (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b' : …
  -/
  congr; ext x; constructor <;> intro H c hc d hd
    /-
      case e_a.h.mp
      a b x : Ordinal.{u_1}
      H : Membership.mem (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b'  …
      c : Ordinal.{u_1}
      hc : LT.lt c b
      d : Ordinal.{u_1}
      hd : LT.lt d a
      ⊢ LT.lt ((c.nmul a).nadd (b.nmul d)) (x.nadd (c.nmul d))
    -/
  · rw [nadd_comm, ← nmul_comm, ← nmul_comm a, ← nmul_comm d]
    /-
      case e_a.h.mp
      a b x : Ordinal.{u_1}
      H : Membership.mem (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b'  …
      c : Ordinal.{u_1}
      hc : LT.lt c b
      d : Ordinal.{u_1}
      hd : LT.lt d a
      ⊢ LT.lt ((d.nmul b).nadd (a.nmul c)) (x.nadd (d.nmul c))
    -/
    exact H _ hd _ hc
    /-
      🎉 no goals
    -/
    /-
      case e_a.h.mpr
      a b x : Ordinal.{u_1}
      H : Membership.mem (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' b → ∀ (b'  …
      c : Ordinal.{u_1}
      hc : LT.lt c a
      d : Ordinal.{u_1}
      hd : LT.lt d b
      ⊢ LT.lt ((c.nmul b).nadd (a.nmul d)) (x.nadd (c.nmul d))
    -/
  · rw [nadd_comm, nmul_comm, nmul_comm c, nmul_comm c]
    /-
      case e_a.h.mpr
      a b x : Ordinal.{u_1}
      H : Membership.mem (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' b → ∀ (b'  …
      c : Ordinal.{u_1}
      hc : LT.lt c a
      d : Ordinal.{u_1}
      hd : LT.lt d b
      ⊢ LT.lt ((d.nmul a).nadd (b.nmul c)) (x.nadd (d.nmul c))
    -/
    exact H _ hd _ hc
    /-
      🎉 no goals
    -/
termination_by (a, b)


@[simp]
theorem nmul_zero (a) : a ⨳ 0 = 0 := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (a.nmul 0) 0
  -/
  rw [← Ordinal.le_zero, nmul_le_iff]
  /-
    a : Ordinal.{u_1}
    ⊢ ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b' : Ordinal.{u_1}), LT.lt b' 0 → LT …
  -/
  exact fun _ _ a ha => (Ordinal.not_lt_zero a ha).elim
  /-
    🎉 no goals
  -/


@[simp]
                                        /-
                                          a : Ordinal.{u_1}
                                          ⊢ Eq (Ordinal.nmul 0 a) 0
                                        -/
theorem zero_nmul (a) : 0 ⨳ a = 0 := by rw [nmul_comm, nmul_zero]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem nmul_one (a : Ordinal) : a ⨳ 1 = a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (a.nmul 1) a
  -/
  rw [nmul]
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (InfSet.sInf (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b' : …
  -/
  convert csInf_Ici
  /-
    case h.e'_2.h.e'_3
    a : Ordinal.{u_1}
    ⊢ Eq (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b' : Ordinal.{u_1 …
  -/
  ext b
  /-
    case h.e'_2.h.e'_3.h
    a b : Ordinal.{u_1}
    ⊢ Iff (Membership.mem (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ ( …
  -/
  refine ⟨fun H ↦ le_of_forall_lt (a := a) fun c hc ↦ ?_, fun ha c hc ↦ ?_⟩
  -- Porting note: had to add arguments to `nmul_one` in the next two lines
  -- for the termination checker.
    /-
      case h.e'_2.h.e'_3.h.refine_1
      a b : Ordinal.{u_1}
      H : Membership.mem (setOf fun c => ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b'  …
      c : Ordinal.{u_1}
      hc : LT.lt c a
      ⊢ LT.lt c b
    -/
  · simpa [nmul_one c] using H c hc
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_3.h.refine_2
      a b : Ordinal.{u_1}
      ha : Membership.mem (Set.Ici a) b
      c : Ordinal.{u_1}
      hc : LT.lt c a
      ⊢ ∀ (b' : Ordinal.{u_1}), LT.lt b' 1 → LT.lt ((c.nmul 1).nadd (a.nmul b')) (b. …
    -/
  · simpa [nmul_one c] using hc.trans_le ha
    /-
      🎉 no goals
    -/
termination_by a


@[simp]
                                       /-
                                         a : Ordinal.{u_1}
                                         ⊢ Eq (Ordinal.nmul 1 a) a
                                       -/
theorem one_nmul (a) : 1 ⨳ a = a := by rw [nmul_comm, nmul_one]
                                       /-
                                         🎉 no goals
                                       -/


theorem nmul_lt_nmul_of_pos_left (h₁ : a < b) (h₂ : 0 < c) : c ⨳ a < c ⨳ b :=
                                  /-
                                    a b c : Ordinal.{u}
                                    h₁ : LT.lt a b
                                    h₂ : LT.lt 0 c
                                    ⊢ LE.le ((c.nmul a).nadd (Ordinal.nmul 0 a)) ((Ordinal.nmul 0 b).nadd (c.nmul  …
                                  -/
  lt_nmul_iff.2 ⟨0, h₂, a, h₁, by simp⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem nmul_lt_nmul_of_pos_right (h₁ : a < b) (h₂ : 0 < c) : a ⨳ c < b ⨳ c :=
                                  /-
                                    a b c : Ordinal.{u}
                                    h₁ : LT.lt a b
                                    h₂ : LT.lt 0 c
                                    ⊢ LE.le ((a.nmul c).nadd (a.nmul 0)) ((a.nmul c).nadd (b.nmul 0))
                                  -/
  lt_nmul_iff.2 ⟨a, h₁, 0, h₂, by simp⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem nmul_le_nmul_left (h : a ≤ b) (c) : c ⨳ a ≤ c ⨳ b := by
  /-
    a b : Ordinal.{u}
    h : LE.le a b
    c : Ordinal.{u}
    ⊢ LE.le (c.nmul a) (c.nmul b)
  -/
  rcases lt_or_eq_of_le h with (h₁ | rfl) <;> rcases (eq_zero_or_pos c).symm with (h₂ | rfl)
    /-
      case inl.inl
      a b : Ordinal.{u}
      h : LE.le a b
      c : Ordinal.{u}
      h₁ : LT.lt a b
      h₂ : LT.lt 0 c
      ⊢ LE.le (c.nmul a) (c.nmul b)
    -/
  · exact (nmul_lt_nmul_of_pos_left h₁ h₂).le
    /-
      🎉 no goals
    -/
  /-
    case inl.inr
    a b : Ordinal.{u}
    h : LE.le a b
    h₁ : LT.lt a b
    ⊢ LE.le (Ordinal.nmul 0 a) (Ordinal.nmul 0 b)
  -/
  all_goals simp
  /-
    🎉 no goals
  -/


@[deprecated nmul_le_nmul_left (since := "2024-08-20")]
alias nmul_le_nmul_of_nonneg_left := nmul_le_nmul_left


theorem nmul_le_nmul_right (h : a ≤ b) (c) : a ⨳ c ≤ b ⨳ c := by
  /-
    a b : Ordinal.{u}
    h : LE.le a b
    c : Ordinal.{u}
    ⊢ LE.le (a.nmul c) (b.nmul c)
  -/
  rw [nmul_comm, nmul_comm b]
  /-
    a b : Ordinal.{u}
    h : LE.le a b
    c : Ordinal.{u}
    ⊢ LE.le (c.nmul a) (c.nmul b)
  -/
  exact nmul_le_nmul_left h c
  /-
    🎉 no goals
  -/


@[deprecated nmul_le_nmul_left (since := "2024-08-20")]
alias nmul_le_nmul_of_nonneg_right := nmul_le_nmul_right


theorem nmul_nadd (a b c : Ordinal) : a ⨳ (b ♯ c) = a ⨳ b ♯ a ⨳ c := by
  refine le_antisymm (nmul_le_iff.2 fun a' ha d hd => ?_)
    (nadd_le_iff.2 ⟨fun d hd => ?_, fun d hd => ?_⟩)
    /-
      case refine_1
      a b c a' : Ordinal.{u_1}
      ha : LT.lt a' a
      d : Ordinal.{u_1}
      hd : LT.lt d (b.nadd c)
      ⊢ LT.lt ((a'.nmul (b.nadd c)).nadd (a.nmul d)) (((a.nmul b).nadd (a.nmul c)).n …
    -/
  · rw [nmul_nadd]
    /-
      case refine_1
      a b c a' : Ordinal.{u_1}
      ha : LT.lt a' a
      d : Ordinal.{u_1}
      hd : LT.lt d (b.nadd c)
      ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
    -/
    rcases lt_nadd_iff.1 hd with (⟨b', hb, hd⟩ | ⟨c', hc, hd⟩)
      /-
        case refine_1.inl.intro.intro
        a b c a' : Ordinal.{u_1}
        ha : LT.lt a' a
        d : Ordinal.{u_1}
        hd✝ : LT.lt d (b.nadd c)
        b' : Ordinal.{u_1}
        hb : LT.lt b' b
        hd : LE.le d (b'.nadd c)
        ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
      -/
    · have := nadd_lt_nadd_of_lt_of_le (nmul_nadd_lt ha hb) (nmul_nadd_le ha.le hd)
      /-
        case refine_1.inl.intro.intro
        a b c a' : Ordinal.{u_1}
        ha : LT.lt a' a
        d : Ordinal.{u_1}
        hd✝ : LT.lt d (b.nadd c)
        b' : Ordinal.{u_1}
        hb : LT.lt b' b
        hd : LE.le d (b'.nadd c)
        this : LT.lt (((a'.nmul b).nadd (a.nmul b')).nadd ((a'.nmul (b'.nadd c)).nadd  …
        ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
      -/
      rw [nmul_nadd, nmul_nadd] at this
      /-
        case refine_1.inl.intro.intro
        a b c a' : Ordinal.{u_1}
        ha : LT.lt a' a
        d : Ordinal.{u_1}
        hd✝ : LT.lt d (b.nadd c)
        b' : Ordinal.{u_1}
        hb : LT.lt b' b
        hd : LE.le d (b'.nadd c)
        this : LT.lt (((a'.nmul b).nadd (a.nmul b')).nadd (((a'.nmul b').nadd (a'.nmul …
        ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
      -/
      simp only [nadd_assoc] at this
      rwa [nadd_left_comm, nadd_left_comm _ (a ⨳ b'), nadd_left_comm (a ⨳ b),
        nadd_lt_nadd_iff_left, nadd_left_comm (a' ⨳ b), nadd_left_comm (a ⨳ b),
        nadd_lt_nadd_iff_left, ← nadd_assoc, ← nadd_assoc] at this
      /-
        case refine_1.inr.intro.intro
        a b c a' : Ordinal.{u_1}
        ha : LT.lt a' a
        d : Ordinal.{u_1}
        hd✝ : LT.lt d (b.nadd c)
        c' : Ordinal.{u_1}
        hc : LT.lt c' c
        hd : LE.le d (b.nadd c')
        ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
      -/
    · have := nadd_lt_nadd_of_le_of_lt (nmul_nadd_le ha.le hd) (nmul_nadd_lt ha hc)
      /-
        case refine_1.inr.intro.intro
        a b c a' : Ordinal.{u_1}
        ha : LT.lt a' a
        d : Ordinal.{u_1}
        hd✝ : LT.lt d (b.nadd c)
        c' : Ordinal.{u_1}
        hc : LT.lt c' c
        hd : LE.le d (b.nadd c')
        this : LT.lt (((a'.nmul (b.nadd c')).nadd (a.nmul d)).nadd ((a'.nmul c).nadd ( …
        ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
      -/
      rw [nmul_nadd, nmul_nadd] at this
      /-
        case refine_1.inr.intro.intro
        a b c a' : Ordinal.{u_1}
        ha : LT.lt a' a
        d : Ordinal.{u_1}
        hd✝ : LT.lt d (b.nadd c)
        c' : Ordinal.{u_1}
        hc : LT.lt c' c
        hd : LE.le d (b.nadd c')
        this : LT.lt ((((a'.nmul b).nadd (a'.nmul c')).nadd (a.nmul d)).nadd ((a'.nmul …
        ⊢ LT.lt (((a'.nmul b).nadd (a'.nmul c)).nadd (a.nmul d)) (((a.nmul b).nadd (a. …
      -/
      simp only [nadd_assoc] at this
      rwa [nadd_left_comm, nadd_comm (a ⨳ c), nadd_left_comm (a' ⨳ d), nadd_left_comm (a ⨳ c'),
        nadd_left_comm (a ⨳ b), nadd_lt_nadd_iff_left, nadd_comm (a' ⨳ c), nadd_left_comm (a ⨳ d),
        nadd_left_comm (a' ⨳ b), nadd_left_comm (a ⨳ b), nadd_lt_nadd_iff_left, nadd_comm (a ⨳ d),
        nadd_comm (a' ⨳ d), ← nadd_assoc, ← nadd_assoc] at this
    /-
      case refine_2
      a b c d : Ordinal.{u_1}
      hd : LT.lt d (a.nmul b)
      ⊢ LT.lt (d.nadd (a.nmul c)) (a.nmul (b.nadd c))
    -/
  · rcases lt_nmul_iff.1 hd with ⟨a', ha, b', hb, hd⟩
    /-
      case refine_2.intro.intro.intro.intro
      a b c d : Ordinal.{u_1}
      hd✝ : LT.lt d (a.nmul b)
      a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      hd : LE.le (d.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      ⊢ LT.lt (d.nadd (a.nmul c)) (a.nmul (b.nadd c))
    -/
    have := nadd_lt_nadd_of_le_of_lt hd (nmul_nadd_lt ha (nadd_lt_nadd_right hb c))
    /-
      case refine_2.intro.intro.intro.intro
      a b c d : Ordinal.{u_1}
      hd✝ : LT.lt d (a.nmul b)
      a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      hd : LE.le (d.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      this : LT.lt ((d.nadd (a'.nmul b')).nadd ((a'.nmul (b.nadd c)).nadd (a.nmul (b …
      ⊢ LT.lt (d.nadd (a.nmul c)) (a.nmul (b.nadd c))
    -/
    rw [nmul_nadd, nmul_nadd, nmul_nadd a'] at this
    /-
      case refine_2.intro.intro.intro.intro
      a b c d : Ordinal.{u_1}
      hd✝ : LT.lt d (a.nmul b)
      a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      hd : LE.le (d.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      this : LT.lt ((d.nadd (a'.nmul b')).nadd (((a'.nmul b).nadd (a'.nmul c)).nadd  …
      ⊢ LT.lt (d.nadd (a.nmul c)) (a.nmul (b.nadd c))
    -/
    simp only [nadd_assoc] at this
    rwa [nadd_left_comm (a' ⨳ b'), nadd_left_comm, nadd_lt_nadd_iff_left, nadd_left_comm,
      nadd_left_comm _ (a' ⨳ b'), nadd_left_comm (a ⨳ b'), nadd_lt_nadd_iff_left,
      nadd_left_comm (a' ⨳ c), nadd_left_comm, nadd_lt_nadd_iff_left, nadd_left_comm,
      nadd_comm _ (a' ⨳ c), nadd_lt_nadd_iff_left] at this
    /-
      case refine_3
      a b c d : Ordinal.{u_1}
      hd : LT.lt d (a.nmul c)
      ⊢ LT.lt ((a.nmul b).nadd d) (a.nmul (b.nadd c))
    -/
  · rcases lt_nmul_iff.1 hd with ⟨a', ha, c', hc, hd⟩
    /-
      case refine_3.intro.intro.intro.intro
      a b c d : Ordinal.{u_1}
      hd✝ : LT.lt d (a.nmul c)
      a' : Ordinal.{u_1}
      ha : LT.lt a' a
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      hd : LE.le (d.nadd (a'.nmul c')) ((a'.nmul c).nadd (a.nmul c'))
      ⊢ LT.lt ((a.nmul b).nadd d) (a.nmul (b.nadd c))
    -/
    have := nadd_lt_nadd_of_lt_of_le (nmul_nadd_lt ha (nadd_lt_nadd_left hc b)) hd
    /-
      case refine_3.intro.intro.intro.intro
      a b c d : Ordinal.{u_1}
      hd✝ : LT.lt d (a.nmul c)
      a' : Ordinal.{u_1}
      ha : LT.lt a' a
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      hd : LE.le (d.nadd (a'.nmul c')) ((a'.nmul c).nadd (a.nmul c'))
      this : LT.lt (((a'.nmul (b.nadd c)).nadd (a.nmul (b.nadd c'))).nadd (d.nadd (a …
      ⊢ LT.lt ((a.nmul b).nadd d) (a.nmul (b.nadd c))
    -/
    rw [nmul_nadd, nmul_nadd, nmul_nadd a'] at this
    /-
      case refine_3.intro.intro.intro.intro
      a b c d : Ordinal.{u_1}
      hd✝ : LT.lt d (a.nmul c)
      a' : Ordinal.{u_1}
      ha : LT.lt a' a
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      hd : LE.le (d.nadd (a'.nmul c')) ((a'.nmul c).nadd (a.nmul c'))
      this : LT.lt ((((a'.nmul b).nadd (a'.nmul c)).nadd ((a.nmul b).nadd (a.nmul c' …
      ⊢ LT.lt ((a.nmul b).nadd d) (a.nmul (b.nadd c))
    -/
    simp only [nadd_assoc] at this
    rwa [nadd_left_comm _ (a' ⨳ b), nadd_lt_nadd_iff_left, nadd_left_comm (a' ⨳ c'),
      nadd_left_comm _ (a' ⨳ c), nadd_lt_nadd_iff_left, nadd_left_comm, nadd_comm (a' ⨳ c'),
      nadd_left_comm _ (a ⨳ c'), nadd_lt_nadd_iff_left, nadd_comm _ (a' ⨳ c'),
      nadd_comm _ (a' ⨳ c'), nadd_left_comm, nadd_lt_nadd_iff_left] at this
termination_by (a, b, c)


theorem nadd_nmul (a b c) : (a ♯ b) ⨳ c = a ⨳ c ♯ b ⨳ c := by
  /-
    a b c : Ordinal.{u_1}
    ⊢ Eq ((a.nadd b).nmul c) ((a.nmul c).nadd (b.nmul c))
  -/
  rw [nmul_comm, nmul_nadd, nmul_comm, nmul_comm c]
  /-
    🎉 no goals
  -/


theorem nmul_nadd_lt₃ {a' b' c' : Ordinal} (ha : a' < a) (hb : b' < b) (hc : c' < c) :
    a' ⨳ b ⨳ c ♯ a ⨳ b' ⨳ c ♯ a ⨳ b ⨳ c' ♯ a' ⨳ b' ⨳ c' <
      a ⨳ b ⨳ c ♯ a' ⨳ b' ⨳ c ♯ a' ⨳ b ⨳ c' ♯ a ⨳ b' ⨳ c' := by
  /-
    a b c a' b' c' : Ordinal.{u}
    ha : LT.lt a' a
    hb : LT.lt b' b
    hc : LT.lt c' c
    ⊢ LT.lt (((((a'.nmul b).nmul c).nadd ((a.nmul b').nmul c)).nadd ((a.nmul b).nm …
  -/
  simpa only [nadd_nmul, ← nadd_assoc] using nmul_nadd_lt (nmul_nadd_lt ha hb) hc
  /-
    🎉 no goals
  -/


theorem nmul_nadd_le₃ {a' b' c' : Ordinal} (ha : a' ≤ a) (hb : b' ≤ b) (hc : c' ≤ c) :
    a' ⨳ b ⨳ c ♯ a ⨳ b' ⨳ c ♯ a ⨳ b ⨳ c' ♯ a' ⨳ b' ⨳ c' ≤
      a ⨳ b ⨳ c ♯ a' ⨳ b' ⨳ c ♯ a' ⨳ b ⨳ c' ♯ a ⨳ b' ⨳ c' := by
  /-
    a b c a' b' c' : Ordinal.{u}
    ha : LE.le a' a
    hb : LE.le b' b
    hc : LE.le c' c
    ⊢ LE.le (((((a'.nmul b).nmul c).nadd ((a.nmul b').nmul c)).nadd ((a.nmul b).nm …
  -/
  simpa only [nadd_nmul, ← nadd_assoc] using nmul_nadd_le (nmul_nadd_le ha hb) hc
  /-
    🎉 no goals
  -/


private theorem nmul_nadd_lt₃' {a' b' c' : Ordinal} (ha : a' < a) (hb : b' < b) (hc : c' < c) :
    a' ⨳ (b ⨳ c) ♯ a ⨳ (b' ⨳ c) ♯ a ⨳ (b ⨳ c') ♯ a' ⨳ (b' ⨳ c') <
      a ⨳ (b ⨳ c) ♯ a' ⨳ (b' ⨳ c) ♯ a' ⨳ (b ⨳ c') ♯ a ⨳ (b' ⨳ c') := by
  /-
    a b c a' b' c' : Ordinal.{u}
    ha : LT.lt a' a
    hb : LT.lt b' b
    hc : LT.lt c' c
    ⊢ LT.lt ((((a'.nmul (b.nmul c)).nadd (a.nmul (b'.nmul c))).nadd (a.nmul (b.nmu …
  -/
  simp only [nmul_comm _ (_ ⨳ _)]
  /-
    a b c a' b' c' : Ordinal.{u}
    ha : LT.lt a' a
    hb : LT.lt b' b
    hc : LT.lt c' c
    ⊢ LT.lt (((((b.nmul c).nmul a').nadd ((b'.nmul c).nmul a)).nadd ((b.nmul c').n …
  -/
  convert nmul_nadd_lt₃ hb hc ha using 1 <;>
     /-
       case h.e'_3
       a b c a' b' c' : Ordinal.{u}
       ha : LT.lt a' a
       hb : LT.lt b' b
       hc : LT.lt c' c
       ⊢ Eq (((((b.nmul c).nmul a').nadd ((b'.nmul c).nmul a)).nadd ((b.nmul c').nmul …
     -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    (simp only [nadd_eq_add, NatOrdinal.toOrdinal_toNatOrdinal]; abel_nf)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated nmul_nadd_le₃ (since := "2024-11-19")]
theorem nmul_nadd_le₃' {a' b' c' : Ordinal} (ha : a' ≤ a) (hb : b' ≤ b) (hc : c' ≤ c) :
    a' ⨳ (b ⨳ c) ♯ a ⨳ (b' ⨳ c) ♯ a ⨳ (b ⨳ c') ♯ a' ⨳ (b' ⨳ c') ≤
      a ⨳ (b ⨳ c) ♯ a' ⨳ (b' ⨳ c) ♯ a' ⨳ (b ⨳ c') ♯ a ⨳ (b' ⨳ c') := by
  /-
    a b c a' b' c' : Ordinal.{u}
    ha : LE.le a' a
    hb : LE.le b' b
    hc : LE.le c' c
    ⊢ LE.le ((((a'.nmul (b.nmul c)).nadd (a.nmul (b'.nmul c))).nadd (a.nmul (b.nmu …
  -/
  simp only [nmul_comm _ (_ ⨳ _)]
  /-
    a b c a' b' c' : Ordinal.{u}
    ha : LE.le a' a
    hb : LE.le b' b
    hc : LE.le c' c
    ⊢ LE.le (((((b.nmul c).nmul a').nadd ((b'.nmul c).nmul a)).nadd ((b.nmul c').n …
  -/
  convert nmul_nadd_le₃ hb hc ha using 1 <;>
     /-
       case h.e'_3
       a b c a' b' c' : Ordinal.{u}
       ha : LE.le a' a
       hb : LE.le b' b
       hc : LE.le c' c
       ⊢ Eq (((((b.nmul c).nmul a').nadd ((b'.nmul c).nmul a)).nadd ((b.nmul c').nmul …
     -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    (simp only [nadd_eq_add, NatOrdinal.toOrdinal_toNatOrdinal]; abel_nf)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem lt_nmul_iff₃ : d < a ⨳ b ⨳ c ↔ ∃ a' < a, ∃ b' < b, ∃ c' < c,
    d ♯ a' ⨳ b' ⨳ c ♯ a' ⨳ b ⨳ c' ♯ a ⨳ b' ⨳ c' ≤
      a' ⨳ b ⨳ c ♯ a ⨳ b' ⨳ c ♯ a ⨳ b ⨳ c' ♯ a' ⨳ b' ⨳ c' := by
  /-
    a b c d : Ordinal.{u}
    ⊢ Iff (LT.lt d ((a.nmul b).nmul c)) (Exists fun a' => And (LT.lt a' a) (Exists …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨a', ha, b', hb, c', hc, h⟩ ↦ ?_⟩
    /-
      case refine_1
      a b c d : Ordinal.{u}
      h : LT.lt d ((a.nmul b).nmul c)
      ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Exists …
    -/
  · rcases lt_nmul_iff.1 h with ⟨e, he, c', hc, H₁⟩
    /-
      case refine_1.intro.intro.intro.intro
      a b c d : Ordinal.{u}
      h : LT.lt d ((a.nmul b).nmul c)
      e : Ordinal.{u}
      he : LT.lt e (a.nmul b)
      c' : Ordinal.{u}
      hc : LT.lt c' c
      H₁ : LE.le (d.nadd (e.nmul c')) ((e.nmul c).nadd ((a.nmul b).nmul c'))
      ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Exists …
    -/
    rcases lt_nmul_iff.1 he with ⟨a', ha, b', hb, H₂⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      a b c d : Ordinal.{u}
      h : LT.lt d ((a.nmul b).nmul c)
      e : Ordinal.{u}
      he : LT.lt e (a.nmul b)
      c' : Ordinal.{u}
      hc : LT.lt c' c
      H₁ : LE.le (d.nadd (e.nmul c')) ((e.nmul c).nadd ((a.nmul b).nmul c'))
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      H₂ : LE.le (e.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      ⊢ Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Exists …
    -/
    refine ⟨a', ha, b', hb, c', hc, ?_⟩
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      a b c d : Ordinal.{u}
      h : LT.lt d ((a.nmul b).nmul c)
      e : Ordinal.{u}
      he : LT.lt e (a.nmul b)
      c' : Ordinal.{u}
      hc : LT.lt c' c
      H₁ : LE.le (d.nadd (e.nmul c')) ((e.nmul c).nadd ((a.nmul b).nmul c'))
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      H₂ : LE.le (e.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      ⊢ LE.le (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd ((a. …
    -/
    have := nadd_le_nadd H₁ (nmul_nadd_le H₂ hc.le)
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      a b c d : Ordinal.{u}
      h : LT.lt d ((a.nmul b).nmul c)
      e : Ordinal.{u}
      he : LT.lt e (a.nmul b)
      c' : Ordinal.{u}
      hc : LT.lt c' c
      H₁ : LE.le (d.nadd (e.nmul c')) ((e.nmul c).nadd ((a.nmul b).nmul c'))
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      H₂ : LE.le (e.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      this : LE.le ((d.nadd (e.nmul c')).nadd (((e.nadd (a'.nmul b')).nmul c).nadd ( …
      ⊢ LE.le (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd ((a. …
    -/
    simp only [nadd_nmul, nadd_assoc] at this
    rw [nadd_left_comm, nadd_left_comm d, nadd_left_comm, nadd_le_nadd_iff_left,
      nadd_left_comm (a ⨳ b' ⨳ c), nadd_left_comm (a' ⨳ b ⨳ c), nadd_left_comm (a ⨳ b ⨳ c'),
      nadd_le_nadd_iff_left, nadd_left_comm (a ⨳ b ⨳ c'), nadd_left_comm (a ⨳ b ⨳ c')] at this
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
      a b c d : Ordinal.{u}
      h : LT.lt d ((a.nmul b).nmul c)
      e : Ordinal.{u}
      he : LT.lt e (a.nmul b)
      c' : Ordinal.{u}
      hc : LT.lt c' c
      H₁ : LE.le (d.nadd (e.nmul c')) ((e.nmul c).nadd ((a.nmul b).nmul c'))
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      H₂ : LE.le (e.nadd (a'.nmul b')) ((a'.nmul b).nadd (a.nmul b'))
      this : LE.le (d.nadd (((a'.nmul b').nmul c).nadd (((a'.nmul b).nmul c').nadd ( …
      ⊢ LE.le (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd ((a. …
    -/
    simpa only [nadd_assoc]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b c d : Ordinal.{u}
      x✝ : Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Exi …
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      c' : Ordinal.{u}
      hc : LT.lt c' c
      h : LE.le (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd (( …
      ⊢ LT.lt d ((a.nmul b).nmul c)
    -/
  · have := h.trans_lt (nmul_nadd_lt₃ ha hb hc)
    /-
      case refine_2
      a b c d : Ordinal.{u}
      x✝ : Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Exi …
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      c' : Ordinal.{u}
      hc : LT.lt c' c
      h : LE.le (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd (( …
      this : LT.lt (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd …
      ⊢ LT.lt d ((a.nmul b).nmul c)
    -/
    repeat rw [nadd_lt_nadd_iff_right] at this
    /-
      case refine_2
      a b c d : Ordinal.{u}
      x✝ : Exists fun a' => And (LT.lt a' a) (Exists fun b' => And (LT.lt b' b) (Exi …
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      c' : Ordinal.{u}
      hc : LT.lt c' c
      h : LE.le (((d.nadd ((a'.nmul b').nmul c)).nadd ((a'.nmul b).nmul c')).nadd (( …
      this : LT.lt d ((a.nmul b).nmul c)
      ⊢ LT.lt d ((a.nmul b).nmul c)
    -/
    assumption
    /-
      🎉 no goals
    -/


theorem nmul_le_iff₃ : a ⨳ b ⨳ c ≤ d ↔ ∀ a' < a, ∀ b' < b, ∀ c' < c,
    a' ⨳ b ⨳ c ♯ a ⨳ b' ⨳ c ♯ a ⨳ b ⨳ c' ♯ a' ⨳ b' ⨳ c' <
      d ♯ a' ⨳ b' ⨳ c ♯ a' ⨳ b ⨳ c' ♯ a ⨳ b' ⨳ c' := by
  /-
    a b c d : Ordinal.{u}
    ⊢ Iff (LE.le ((a.nmul b).nmul c) d) (∀ (a' : Ordinal.{u}), LT.lt a' a → ∀ (b'  …
  -/
  simpa using lt_nmul_iff₃.not
  /-
    🎉 no goals
  -/


private theorem nmul_le_iff₃' : a ⨳ (b ⨳ c) ≤ d ↔ ∀ a' < a, ∀ b' < b, ∀ c' < c,
    a' ⨳ (b ⨳ c) ♯ a ⨳ (b' ⨳ c) ♯ a ⨳ (b ⨳ c') ♯ a' ⨳ (b' ⨳ c') <
      d ♯ a' ⨳ (b' ⨳ c) ♯ a' ⨳ (b ⨳ c') ♯ a ⨳ (b' ⨳ c') := by
  /-
    a b c d : Ordinal.{u}
    ⊢ Iff (LE.le (a.nmul (b.nmul c)) d) (∀ (a' : Ordinal.{u}), LT.lt a' a → ∀ (b'  …
  -/
  simp only [nmul_comm _ (_ ⨳ _), nmul_le_iff₃, nadd_eq_add, toOrdinal_toNatOrdinal]
  /-
    a b c d : Ordinal.{u}
    ⊢ Iff (∀ (a' : Ordinal.{u}), LT.lt a' b → ∀ (b' : Ordinal.{u}), LT.lt b' c → ∀ …
  -/
  constructor <;> intro h a' ha b' hb c' hc
    /-
      case mp
      a b c d : Ordinal.{u}
      h : ∀ (a' : Ordinal.{u}), LT.lt a' b → ∀ (b' : Ordinal.{u}), LT.lt b' c → ∀ (c …
      a' : Ordinal.{u}
      ha : LT.lt a' a
      b' : Ordinal.{u}
      hb : LT.lt b' b
      c' : Ordinal.{u}
      hc : LT.lt c' c
      ⊢ LT.lt (NatOrdinal.toOrdinal (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Ordinal.toNatO …
    -/
                                            /-
                                              🎉 no goals
                                            -/
  · convert h b' hb c' hc a' ha using 1 <;> abel_nf
                                            /-
                                              🎉 no goals
                                            -/
    /-
      case mpr
      a b c d : Ordinal.{u}
      h : ∀ (a' : Ordinal.{u}), LT.lt a' a → ∀ (b' : Ordinal.{u}), LT.lt b' b → ∀ (c …
      a' : Ordinal.{u}
      ha : LT.lt a' b
      b' : Ordinal.{u}
      hb : LT.lt b' c
      c' : Ordinal.{u}
      hc : LT.lt c' a
      ⊢ LT.lt (NatOrdinal.toOrdinal (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Ordinal.toNatO …
    -/
                                            /-
                                              🎉 no goals
                                            -/
  · convert h c' hc a' ha b' hb using 1 <;> abel_nf
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated lt_nmul_iff₃ (since := "2024-11-19")]
theorem lt_nmul_iff₃' : d < a ⨳ (b ⨳ c) ↔ ∃ a' < a, ∃ b' < b, ∃ c' < c,
    d ♯ a' ⨳ (b' ⨳ c) ♯ a' ⨳ (b ⨳ c') ♯ a ⨳ (b' ⨳ c') ≤
      a' ⨳ (b ⨳ c) ♯ a ⨳ (b' ⨳ c) ♯ a ⨳ (b ⨳ c') ♯ a' ⨳ (b' ⨳ c') := by
  /-
    a b c d : Ordinal.{u}
    ⊢ Iff (LT.lt d (a.nmul (b.nmul c))) (Exists fun a' => And (LT.lt a' a) (Exists …
  -/
  simpa using nmul_le_iff₃'.not
  /-
    🎉 no goals
  -/


theorem nmul_assoc (a b c : Ordinal) : a ⨳ b ⨳ c = a ⨳ (b ⨳ c) := by
  /-
    a b c : Ordinal.{u_1}
    ⊢ Eq ((a.nmul b).nmul c) (a.nmul (b.nmul c))
  -/
  apply le_antisymm
    /-
      case a
      a b c : Ordinal.{u_1}
      ⊢ LE.le ((a.nmul b).nmul c) (a.nmul (b.nmul c))
    -/
  · rw [nmul_le_iff₃]
    /-
      case a
      a b c : Ordinal.{u_1}
      ⊢ ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b' : Ordinal.{u_1}), LT.lt b' b → ∀  …
    -/
    intro a' ha b' hb c' hc
    /-
      case a
      a b c a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      ⊢ LT.lt (((((a'.nmul b).nmul c).nadd ((a.nmul b').nmul c)).nadd ((a.nmul b).nm …
    -/
    repeat rw [nmul_assoc]
    /-
      case a
      a b c a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      ⊢ LT.lt ((((a'.nmul (b.nmul c)).nadd (a.nmul (b'.nmul c))).nadd (a.nmul (b.nmu …
    -/
    exact nmul_nadd_lt₃' ha hb hc
    /-
      🎉 no goals
    -/
    /-
      case a
      a b c : Ordinal.{u_1}
      ⊢ LE.le (a.nmul (b.nmul c)) ((a.nmul b).nmul c)
    -/
  · rw [nmul_le_iff₃']
    /-
      case a
      a b c : Ordinal.{u_1}
      ⊢ ∀ (a' : Ordinal.{u_1}), LT.lt a' a → ∀ (b' : Ordinal.{u_1}), LT.lt b' b → ∀  …
    -/
    intro a' ha b' hb c' hc
    /-
      case a
      a b c a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      ⊢ LT.lt ((((a'.nmul (b.nmul c)).nadd (a.nmul (b'.nmul c))).nadd (a.nmul (b.nmu …
    -/
    repeat rw [← nmul_assoc]
    /-
      case a
      a b c a' : Ordinal.{u_1}
      ha : LT.lt a' a
      b' : Ordinal.{u_1}
      hb : LT.lt b' b
      c' : Ordinal.{u_1}
      hc : LT.lt c' c
      ⊢ LT.lt (((((a'.nmul b).nmul c).nadd ((a.nmul b').nmul c)).nadd ((a.nmul b).nm …
    -/
    exact nmul_nadd_lt₃ ha hb hc
    /-
      🎉 no goals
    -/
termination_by (a, b, c)


instance : Mul NatOrdinal :=
  ⟨nmul⟩

-- Porting note: had to add universe annotations to ensure that the
-- two sources lived in the same universe.

instance : OrderedCommSemiring NatOrdinal.{u} :=
  { NatOrdinal.instOrderedCancelAddCommMonoid.{u},
    NatOrdinal.instLinearOrder.{u} with
    mul := (· * ·)
    left_distrib := nmul_nadd
    right_distrib := nadd_nmul
    zero_mul := zero_nmul
    mul_zero := nmul_zero
    mul_assoc := nmul_assoc
    one := 1
    one_mul := one_nmul
    mul_one := nmul_one
    mul_comm := nmul_comm
    zero_le_one := @zero_le_one Ordinal _ _ _ _
    mul_le_mul_of_nonneg_left := fun _ _ c h _ => nmul_le_nmul_left h c
    mul_le_mul_of_nonneg_right := fun _ _ c h _ => nmul_le_nmul_right h c }


theorem nmul_eq_mul (a b) : a ⨳ b = toOrdinal (toNatOrdinal a * toNatOrdinal b) :=
  rfl


theorem nmul_nadd_one : ∀ a b, a ⨳ (b ♯ 1) = a ⨳ b ♯ a :=
  @mul_add_one NatOrdinal _ _ _


theorem nadd_one_nmul : ∀ a b, (a ♯ 1) ⨳ b = a ⨳ b ♯ b :=
  @add_one_mul NatOrdinal _ _ _


                                                       /-
                                                         a b : Ordinal.{u_1}
                                                         ⊢ Eq (a.nmul (Order.succ b)) ((a.nmul b).nadd a)
                                                       -/
theorem nmul_succ (a b) : a ⨳ succ b = a ⨳ b ♯ a := by rw [← nadd_one, nmul_nadd_one]
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                       /-
                                                         a b : Ordinal.{u_1}
                                                         ⊢ Eq ((Order.succ a).nmul b) ((a.nmul b).nadd b)
                                                       -/
theorem succ_nmul (a b) : succ a ⨳ b = a ⨳ b ♯ b := by rw [← nadd_one, nadd_one_nmul]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem nmul_add_one : ∀ a b, a ⨳ (b + 1) = a ⨳ b ♯ a :=
  nmul_succ


theorem add_one_nmul : ∀ a b, (a + 1) ⨳ b = a ⨳ b ♯ b :=
  succ_nmul


theorem mul_le_nmul (a b : Ordinal.{u}) : a * b ≤ a ⨳ b := by
  /-
    a b : Ordinal.{u}
    ⊢ LE.le (HMul.hMul a b) (a.nmul b)
  -/
  refine b.limitRecOn ?_ ?_ ?_
    /-
      case refine_1
      a b : Ordinal.{u}
      ⊢ LE.le (HMul.hMul a 0) (a.nmul 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Ordinal.{u}
      ⊢ ∀ (o : Ordinal.{u}), LE.le (HMul.hMul a o) (a.nmul o) → LE.le (HMul.hMul a ( …
    -/
  · intro c h
    /-
      case refine_2
      a b c : Ordinal.{u}
      h : LE.le (HMul.hMul a c) (a.nmul c)
      ⊢ LE.le (HMul.hMul a (Order.succ c)) (a.nmul (Order.succ c))
    -/
    rw [mul_succ, nmul_succ]
    /-
      case refine_2
      a b c : Ordinal.{u}
      h : LE.le (HMul.hMul a c) (a.nmul c)
      ⊢ LE.le (HAdd.hAdd (HMul.hMul a c) a) ((a.nmul c).nadd a)
    -/
    exact (add_le_nadd _ a).trans (nadd_le_nadd_right h a)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      a b : Ordinal.{u}
      ⊢ ∀ (o : Ordinal.{u}), o.IsLimit → (∀ (o' : Ordinal.{u}), LT.lt o' o → LE.le ( …
    -/
  · intro c hc H
    /-
      case refine_3
      a b c : Ordinal.{u}
      hc : c.IsLimit
      H : ∀ (o' : Ordinal.{u}), LT.lt o' c → LE.le (HMul.hMul a o') (a.nmul o')
      ⊢ LE.le (HMul.hMul a c) (a.nmul c)
    -/
    rcases eq_zero_or_pos a with (rfl | ha)
      /-
        case refine_3.inl
        b c : Ordinal.{u}
        hc : c.IsLimit
        H : ∀ (o' : Ordinal.{u}), LT.lt o' c → LE.le (HMul.hMul 0 o') (Ordinal.nmul 0  …
        ⊢ LE.le (HMul.hMul 0 c) (Ordinal.nmul 0 c)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_3.inr
        a b c : Ordinal.{u}
        hc : c.IsLimit
        H : ∀ (o' : Ordinal.{u}), LT.lt o' c → LE.le (HMul.hMul a o') (a.nmul o')
        ha : LT.lt 0 a
        ⊢ LE.le (HMul.hMul a c) (a.nmul c)
      -/
    · rw [← IsNormal.blsub_eq.{u, u} (isNormal_mul_right ha) hc, blsub_le_iff]
      /-
        case refine_3.inr
        a b c : Ordinal.{u}
        hc : c.IsLimit
        H : ∀ (o' : Ordinal.{u}), LT.lt o' c → LE.le (HMul.hMul a o') (a.nmul o')
        ha : LT.lt 0 a
        ⊢ ∀ (i : Ordinal.{u}), LT.lt i c → LT.lt (HMul.hMul a i) (a.nmul c)
      -/
      exact fun i hi => (H i hi).trans_lt (nmul_lt_nmul_of_pos_left hi ha)
      /-
        🎉 no goals
      -/


@[deprecated mul_le_nmul (since := "2024-08-20")]
alias _root_.NatOrdinal.mul_le_nmul := mul_le_nmul


