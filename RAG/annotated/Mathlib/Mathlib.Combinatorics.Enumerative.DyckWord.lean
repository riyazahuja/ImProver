/-- A `DyckStep` is either `U` or `D`, corresponding to `(` and `)` respectively. -/
inductive DyckStep
  | U : DyckStep
  | D : DyckStep
  deriving Inhabited, DecidableEq


/-- Named in analogy to `Bool.dichotomy`. -/
                                                              /-
                                                                s : DyckStep
                                                                ⊢ Or (Eq s DyckStep.U) (Eq s DyckStep.D)
                                                              -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
lemma DyckStep.dichotomy (s : DyckStep) : s = U ∨ s = D := by cases s <;> tauto
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- A Dyck word is a list of `DyckStep`s with as many `U`s as `D`s and with every prefix having
at least as many `U`s as `D`s. -/
@[ext]
structure DyckWord where
  /-- The underlying list -/
  toList : List DyckStep
  /-- There are as many `U`s as `D`s -/
  count_U_eq_count_D : toList.count U = toList.count D
  /-- Each prefix has as least as many `U`s as `D`s -/
  count_D_le_count_U i : (toList.take i).count D ≤ (toList.take i).count U
  deriving DecidableEq


instance : Coe DyckWord (List DyckStep) := ⟨DyckWord.toList⟩


instance : Add DyckWord where
  add p q := ⟨p ++ q, by
    /-
      p q : DyckWord
      ⊢ Eq (List.count DyckStep.U (HAppend.hAppend ↑p ↑q)) (List.count DyckStep.D (H …
    -/
    simp only [count_append, p.count_U_eq_count_D, q.count_U_eq_count_D], by
    /-
      🎉 no goals
    -/
    /-
      p q : DyckWord
      ⊢ ∀ (i : Nat), LE.le (List.count DyckStep.D (List.take i (HAppend.hAppend ↑p ↑ …
    -/
    simp only [take_append_eq_append_take, count_append]
    /-
      p q : DyckWord
      ⊢ ∀ (i : Nat), LE.le (HAdd.hAdd (List.count DyckStep.D (List.take i ↑p)) (List …
    -/
    exact fun _ ↦ add_le_add (p.count_D_le_count_U _) (q.count_D_le_count_U _)⟩
    /-
      🎉 no goals
    -/


                                    /-
                                      ⊢ Eq (List.count DyckStep.U List.nil) (List.count DyckStep.D List.nil)
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
instance : Zero DyckWord := ⟨[], by simp, by simp⟩
                                             /-
                                               🎉 no goals
                                             -/


/-- Dyck words form an additive cancellative monoid under concatenation,
with the empty word as 0. -/
instance : AddCancelMonoid DyckWord where
                   /-
                     p : DyckWord
                     ⊢ Eq (HAdd.hAdd p 0) p
                   -/
                   /-
                     p : DyckWord
                     ⊢ Eq (HAdd.hAdd 0 p) p
                   -/
                        /-
                          p q r : DyckWord
                          ⊢ Eq (HAdd.hAdd (HAdd.hAdd p q) r) (HAdd.hAdd p (HAdd.hAdd q r))
                        -/
  add_zero p := by ext1; exact append_nil _
                              /-
                                🎉 no goals
                              -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
  zero_add p := by ext1; rfl
  add_assoc p q r := by ext1; apply append_assoc
  nsmul := nsmulRec
                                /-
                                  p q r : DyckWord
                                  h : Eq (HAdd.hAdd p q) (HAdd.hAdd p r)
                                  ⊢ Eq q r
                                -/
  add_left_cancel p q r h := by rw [DyckWord.ext_iff] at *; exact append_cancel_left h
                                                            /-
                                                              🎉 no goals
                                                            -/
                                 /-
                                   p q r : DyckWord
                                   h : Eq (HAdd.hAdd p q) (HAdd.hAdd r q)
                                   ⊢ Eq p r
                                 -/
  add_right_cancel p q r h := by rw [DyckWord.ext_iff] at *; exact append_cancel_right h
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                  /-
                                                    p : DyckWord
                                                    ⊢ Iff (Eq (↑p) List.nil) (Eq p 0)
                                                  -/
lemma toList_eq_nil : p.toList = [] ↔ p = 0 := by rw [DyckWord.ext_iff]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

lemma toList_ne_nil : p.toList ≠ [] ↔ p ≠ 0 := toList_eq_nil.ne


/-- The only Dyck word that is an additive unit is the empty word. -/
instance : Unique (AddUnits DyckWord) where
  uniq p := by
    /-
      p✝ q : DyckWord
      p : AddUnits DyckWord
      ⊢ Eq p Inhabited.default
    -/
    obtain ⟨a, b, h, -⟩ := p
    /-
      case mk
      p q a b : DyckWord
      h : Eq (HAdd.hAdd a b) 0
      neg_val✝ : Eq (HAdd.hAdd b a) 0
      ⊢ Eq { val := a, neg := b, val_neg := h, neg_val := neg_val✝ } Inhabited.default
    -/
    obtain ⟨ha, hb⟩ := append_eq_nil.mp (toList_eq_nil.mpr h)
    /-
      case mk.intro
      p q a b : DyckWord
      h : Eq (HAdd.hAdd a b) 0
      neg_val✝ : Eq (HAdd.hAdd b a) 0
      ha : Eq (↑a) List.nil
      hb : Eq (↑b) List.nil
      ⊢ Eq { val := a, neg := b, val_neg := h, neg_val := neg_val✝ } Inhabited.default
    -/
    congr
      /-
        case mk.intro.e_val
        p q a b : DyckWord
        h : Eq (HAdd.hAdd a b) 0
        neg_val✝ : Eq (HAdd.hAdd b a) 0
        ha : Eq (↑a) List.nil
        hb : Eq (↑b) List.nil
        ⊢ Eq a 0
      -/
    · exact toList_eq_nil.mp ha
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.e_neg
        p q a b : DyckWord
        h : Eq (HAdd.hAdd a b) 0
        neg_val✝ : Eq (HAdd.hAdd b a) 0
        ha : Eq (↑a) List.nil
        hb : Eq (↑b) List.nil
        ⊢ Eq b 0
      -/
    · exact toList_eq_nil.mp hb
      /-
        🎉 no goals
      -/


/-- The first element of a nonempty Dyck word is `U`. -/
lemma head_eq_U (p : DyckWord) (h) : p.toList.head h = U := by
  /-
    p : DyckWord
    h : Ne (↑p) List.nil
    ⊢ Eq ((↑p).head h) DyckStep.U
  -/
  rcases p with - | s; · tauto
                         /-
                           🎉 no goals
                         -/
  /-
    case mk.cons
    s : DyckStep
    tail✝ : List DyckStep
    count_U_eq_count_D✝ : Eq (List.count DyckStep.U (List.cons s tail✝)) (List.cou …
    count_D_le_count_U✝ : ∀ (i : Nat), LE.le (List.count DyckStep.D (List.take i ( …
    h : Ne (↑{ toList := List.cons s tail✝, count_U_eq_count_D := count_U_eq_count …
    ⊢ Eq ((↑{ toList := List.cons s tail✝, count_U_eq_count_D := count_U_eq_count_ …
  -/
  rw [head_cons]
  /-
    case mk.cons
    s : DyckStep
    tail✝ : List DyckStep
    count_U_eq_count_D✝ : Eq (List.count DyckStep.U (List.cons s tail✝)) (List.cou …
    count_D_le_count_U✝ : ∀ (i : Nat), LE.le (List.count DyckStep.D (List.take i ( …
    h : Ne (↑{ toList := List.cons s tail✝, count_U_eq_count_D := count_U_eq_count …
    ⊢ Eq s DyckStep.U
  -/
  by_contra f
  /-
    case mk.cons
    s : DyckStep
    tail✝ : List DyckStep
    count_U_eq_count_D✝ : Eq (List.count DyckStep.U (List.cons s tail✝)) (List.cou …
    count_D_le_count_U✝ : ∀ (i : Nat), LE.le (List.count DyckStep.D (List.take i ( …
    h : Ne (↑{ toList := List.cons s tail✝, count_U_eq_count_D := count_U_eq_count …
    f : Not (Eq s DyckStep.U)
    ⊢ False
  -/
  rename_i _ nonneg
  /-
    case mk.cons
    s : DyckStep
    tail✝ : List DyckStep
    count_U_eq_count_D✝ : Eq (List.count DyckStep.U (List.cons s tail✝)) (List.cou …
    nonneg : ∀ (i : Nat), LE.le (List.count DyckStep.D (List.take i (List.cons s t …
    h : Ne (↑{ toList := List.cons s tail✝, count_U_eq_count_D := count_U_eq_count …
    f : Not (Eq s DyckStep.U)
    ⊢ False
  -/
  simpa [s.dichotomy.resolve_left f] using nonneg 1
  /-
    🎉 no goals
  -/


/-- The last element of a nonempty Dyck word is `D`. -/
lemma getLast_eq_D (p : DyckWord) (h) : p.toList.getLast h = D := by
  /-
    p : DyckWord
    h : Ne (↑p) List.nil
    ⊢ Eq ((↑p).getLast h) DyckStep.D
  -/
  by_contra f; have s := p.count_U_eq_count_D
  /-
    p : DyckWord
    h : Ne (↑p) List.nil
    f : Not (Eq ((↑p).getLast h) DyckStep.D)
    s : Eq (List.count DyckStep.U ↑p) (List.count DyckStep.D ↑p)
    ⊢ False
  -/
  rw [← dropLast_append_getLast h, (dichotomy _).resolve_right f] at s
  /-
    p : DyckWord
    h : Ne (↑p) List.nil
    f : Not (Eq ((↑p).getLast h) DyckStep.D)
    s : Eq (List.count DyckStep.U (HAppend.hAppend (↑p).dropLast (List.cons DyckSt …
    ⊢ False
  -/
  simp_rw [dropLast_eq_take, count_append, count_singleton', ite_true, reduceCtorEq, ite_false] at s
  /-
    p : DyckWord
    h : Ne (↑p) List.nil
    f : Not (Eq ((↑p).getLast h) DyckStep.D)
    s : Eq (HAdd.hAdd (List.count DyckStep.U (List.take (HSub.hSub (↑p).length 1)  …
    ⊢ False
  -/
  have := p.count_D_le_count_U (p.toList.length - 1); omega
                                                      /-
                                                        🎉 no goals
                                                      -/


include h in
lemma cons_tail_dropLast_concat : U :: p.toList.dropLast.tail ++ [D] = p := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ Eq (HAppend.hAppend (List.cons DyckStep.U (↑p).dropLast.tail) (List.cons Dyc …
  -/
  have h' := toList_ne_nil.mpr h
  have : p.toList.dropLast.take 1 = [p.toList.head h'] := by
    rcases p with - | ⟨s, ⟨- | ⟨t, r⟩⟩⟩
    · tauto
    · rename_i bal _
      cases s <;> simp at bal
    · tauto
  /-
    p : DyckWord
    h : Ne p 0
    h' : Ne (↑p) List.nil
    this : Eq (List.take 1 (↑p).dropLast) (List.cons ((↑p).head h') List.nil)
    ⊢ Eq (HAppend.hAppend (List.cons DyckStep.U (↑p).dropLast.tail) (List.cons Dyc …
  -/
  nth_rw 2 [← p.toList.dropLast_append_getLast h', ← p.toList.dropLast.take_append_drop 1]
  /-
    p : DyckWord
    h : Ne p 0
    h' : Ne (↑p) List.nil
    this : Eq (List.take 1 (↑p).dropLast) (List.cons ((↑p).head h') List.nil)
    ⊢ Eq (HAppend.hAppend (List.cons DyckStep.U (↑p).dropLast.tail) (List.cons Dyc …
  -/
  rw [getLast_eq_D, drop_one, this, head_eq_U]
  /-
    p : DyckWord
    h : Ne p 0
    h' : Ne (↑p) List.nil
    this : Eq (List.take 1 (↑p).dropLast) (List.cons ((↑p).head h') List.nil)
    ⊢ Eq (HAppend.hAppend (List.cons DyckStep.U (↑p).dropLast.tail) (List.cons Dyc …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable (p) in
/-- Prefix of a Dyck word as a Dyck word, given that the count of `U`s and `D`s in it are equal. -/
def take (i : ℕ) (hi : (p.toList.take i).count U = (p.toList.take i).count D) : DyckWord where
  toList := p.toList.take i
  count_U_eq_count_D := hi
                             /-
                               p q : DyckWord
                               h : Ne p 0
                               i : Nat
                               hi : Eq (List.count DyckStep.U (List.take i ↑p)) (List.count DyckStep.D (List. …
                               k : Nat
                               ⊢ LE.le (List.count DyckStep.D (List.take k (List.take i ↑p))) (List.count Dyc …
                             -/
  count_D_le_count_U k := by rw [take_take]; exact p.count_D_le_count_U (min k i)
                                             /-
                                               🎉 no goals
                                             -/


variable (p) in
/-- Suffix of a Dyck word as a Dyck word, given that the count of `U`s and `D`s in the prefix
are equal. -/
def drop (i : ℕ) (hi : (p.toList.take i).count U = (p.toList.take i).count D) : DyckWord where
  toList := p.toList.drop i
  count_U_eq_count_D := by
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : Eq (List.count DyckStep.U (List.take i ↑p)) (List.count DyckStep.D (List. …
      ⊢ Eq (List.count DyckStep.U (List.drop i ↑p)) (List.count DyckStep.D (List.dro …
    -/
    have := p.count_U_eq_count_D
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : Eq (List.count DyckStep.U (List.take i ↑p)) (List.count DyckStep.D (List. …
      this : Eq (List.count DyckStep.U ↑p) (List.count DyckStep.D ↑p)
      ⊢ Eq (List.count DyckStep.U (List.drop i ↑p)) (List.count DyckStep.D (List.dro …
    -/
    rw [← take_append_drop i p.toList, count_append, count_append] at this
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : Eq (List.count DyckStep.U (List.take i ↑p)) (List.count DyckStep.D (List. …
      this : Eq (HAdd.hAdd (List.count DyckStep.U (List.take i ↑p)) (List.count Dyck …
      ⊢ Eq (List.count DyckStep.U (List.drop i ↑p)) (List.count DyckStep.D (List.dro …
    -/
    omega
    /-
      🎉 no goals
    -/
  count_D_le_count_U k := by
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : Eq (List.count DyckStep.U (List.take i ↑p)) (List.count DyckStep.D (List. …
      k : Nat
      ⊢ LE.le (List.count DyckStep.D (List.take k (List.drop i ↑p))) (List.count Dyc …
    -/
    rw [show i = min i (i + k) by omega, ← take_take] at hi
    rw [take_drop, ← add_le_add_iff_left (((p.toList.take (i + k)).take i).count U),
      ← count_append, hi, ← count_append, take_append_drop]
    /-
      p q : DyckWord
      h : Ne p 0
      i k : Nat
      hi : Eq (List.count DyckStep.U (List.take i (List.take (HAdd.hAdd i k) ↑p))) ( …
      ⊢ LE.le (List.count DyckStep.D (List.take (HAdd.hAdd i k) ↑p)) (List.count Dyc …
    -/
    exact p.count_D_le_count_U _
    /-
      🎉 no goals
    -/


variable (p) in
/-- Nest `p` in one pair of brackets, i.e. `x` becomes `(x)`. -/
def nest : DyckWord where
  toList := [U] ++ p ++ [D]
                           /-
                             p q : DyckWord
                             h : Ne p 0
                             ⊢ Eq (List.count DyckStep.U (HAppend.hAppend (HAppend.hAppend (List.cons DyckS …
                           -/
  count_U_eq_count_D := by simp [p.count_U_eq_count_D]
                           /-
                             🎉 no goals
                           -/
  count_D_le_count_U i := by
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      ⊢ LE.le (List.count DyckStep.D (List.take i (HAppend.hAppend (HAppend.hAppend  …
    -/
    simp only [take_append_eq_append_take, count_append]
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (List.count DyckStep.D (List.take i (List.cons D …
    -/
    rw [← add_rotate (count D _), ← add_rotate (count U _)]
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (List.count DyckStep.D (List.take (HSub.hSub i ( …
    -/
    apply add_le_add _ (p.count_D_le_count_U _)
    /-
      p q : DyckWord
      h : Ne p 0
      i : Nat
      ⊢ LE.le (HAdd.hAdd (List.count DyckStep.D (List.take (HSub.hSub i (HAppend.hAp …
    -/
    rcases i.eq_zero_or_pos with hi | hi; · simp [hi]
                                            /-
                                              🎉 no goals
                                            -/
    /-
      case inr
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : GT.gt i 0
      ⊢ LE.le (HAdd.hAdd (List.count DyckStep.D (List.take (HSub.hSub i (HAppend.hAp …
    -/
    rw [take_of_length_le (show [U].length ≤ i by rwa [length_singleton]), count_singleton']
    /-
      case inr
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : GT.gt i 0
      ⊢ LE.le (HAdd.hAdd (List.count DyckStep.D (List.take (HSub.hSub i (HAppend.hAp …
    -/
    simp only [reduceCtorEq, ite_true, ite_false]
    /-
      case inr
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : GT.gt i 0
      ⊢ LE.le (HAdd.hAdd (List.count DyckStep.D (List.take (HSub.hSub i (HAppend.hAp …
    -/
    rw [add_comm]
    /-
      case inr
      p q : DyckWord
      h : Ne p 0
      i : Nat
      hi : GT.gt i 0
      ⊢ LE.le (HAdd.hAdd 0 (List.count DyckStep.D (List.take (HSub.hSub i (HAppend.h …
    -/
    exact add_le_add (zero_le _) ((count_le_length _ _).trans (by simp))
    /-
      🎉 no goals
    -/


                                              /-
                                                p : DyckWord
                                                ⊢ Ne p.nest 0
                                              -/
@[simp] lemma nest_ne_zero : p.nest ≠ 0 := by simp [← toList_ne_nil, nest]
                                              /-
                                                🎉 no goals
                                              -/


variable (p) in
/-- A property stating that `p` is nonempty and strictly positive in its interior,
i.e. is of the form `(x)` with `x` a Dyck word. -/
def IsNested : Prop :=
  p ≠ 0 ∧ ∀ ⦃i⦄, 0 < i → i < p.toList.length → (p.toList.take i).count D < (p.toList.take i).count U


protected lemma IsNested.nest : p.nest.IsNested := ⟨nest_ne_zero, fun i lb ub ↦ by
  /-
    p : DyckWord
    i : Nat
    lb : LT.lt 0 i
    ub : LT.lt i (↑p.nest).length
    ⊢ LT.lt (List.count DyckStep.D (List.take i ↑p.nest)) (List.count DyckStep.U ( …
  -/
  simp_rw [nest, length_append, length_singleton] at ub ⊢
  rw [take_append_of_le_length (by rw [singleton_append, length_cons]; omega),
    take_append_eq_append_take, take_of_length_le (by rw [length_singleton]; omega),
    length_singleton, singleton_append, count_cons_of_ne (by simp), count_cons_self,
    Nat.lt_add_one_iff]
  /-
    p : DyckWord
    i : Nat
    lb : LT.lt 0 i
    ub : LT.lt i (HAdd.hAdd (HAdd.hAdd 1 (↑p).length) 1)
    ⊢ LE.le (List.count DyckStep.D (List.take (HSub.hSub i 1) ↑p)) (List.count Dyc …
  -/
  exact p.count_D_le_count_U _⟩
  /-
    🎉 no goals
  -/


variable (p) in
/-- Denest `p`, i.e. `(x)` becomes `x`, given that `p.IsNested`. -/
def denest (hn : p.IsNested) : DyckWord where
  toList := p.toList.dropLast.tail
  count_U_eq_count_D := by
    /-
      p q : DyckWord
      h : Ne p 0
      hn : p.IsNested
      ⊢ Eq (List.count DyckStep.U (↑p).dropLast.tail) (List.count DyckStep.D (↑p).dr …
    -/
    have := p.count_U_eq_count_D
    /-
      p q : DyckWord
      h : Ne p 0
      hn : p.IsNested
      this : Eq (List.count DyckStep.U ↑p) (List.count DyckStep.D ↑p)
      ⊢ Eq (List.count DyckStep.U (↑p).dropLast.tail) (List.count DyckStep.D (↑p).dr …
    -/
    rw [← cons_tail_dropLast_concat hn.1, count_append, count_cons] at this
    /-
      p q : DyckWord
      h : Ne p 0
      hn : p.IsNested
      this : Eq (HAdd.hAdd (HAdd.hAdd (List.count DyckStep.U (↑p).dropLast.tail) (it …
      ⊢ Eq (List.count DyckStep.U (↑p).dropLast.tail) (List.count DyckStep.D (↑p).dr …
    -/
    simpa using this
    /-
      🎉 no goals
    -/
  count_D_le_count_U i := by
    /-
      p q : DyckWord
      h : Ne p 0
      hn : p.IsNested
      i : Nat
      ⊢ LE.le (List.count DyckStep.D (List.take i (↑p).dropLast.tail)) (List.count D …
    -/
    replace h := toList_ne_nil.mpr hn.1
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      ⊢ LE.le (List.count DyckStep.D (List.take i (↑p).dropLast.tail)) (List.count D …
    -/
    have l1 : p.toList.take 1 = [p.toList.head h] := by rcases p with - | - <;> tauto
    have l3 : p.toList.length - 1 = p.toList.length - 1 - 1 + 1 := by
      rcases p with - | ⟨s, ⟨- | ⟨t, r⟩⟩⟩
      · tauto
      · rename_i bal _
        cases s <;> simp at bal
      · tauto
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      l1 : Eq (List.take 1 ↑p) (List.cons ((↑p).head h) List.nil)
      l3 : Eq (HSub.hSub (↑p).length 1) (HAdd.hAdd (HSub.hSub (HSub.hSub (↑p).length …
      ⊢ LE.le (List.count DyckStep.D (List.take i (↑p).dropLast.tail)) (List.count D …
    -/
    rw [← drop_one, take_drop, dropLast_eq_take, take_take]
    have ub : min (1 + i) (p.toList.length - 1) < p.toList.length :=
      (min_le_right _ p.toList.length.pred).trans_lt (Nat.pred_lt ((length_pos.mpr h).ne'))
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      l1 : Eq (List.take 1 ↑p) (List.cons ((↑p).head h) List.nil)
      l3 : Eq (HSub.hSub (↑p).length 1) (HAdd.hAdd (HSub.hSub (HSub.hSub (↑p).length …
      ub : LT.lt (Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1)) (↑p).length
      ⊢ LE.le (List.count DyckStep.D (List.drop 1 (List.take (Min.min (HAdd.hAdd 1 i …
    -/
    have lb : 0 < min (1 + i) (p.toList.length - 1) := by omega
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      l1 : Eq (List.take 1 ↑p) (List.cons ((↑p).head h) List.nil)
      l3 : Eq (HSub.hSub (↑p).length 1) (HAdd.hAdd (HSub.hSub (HSub.hSub (↑p).length …
      ub : LT.lt (Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1)) (↑p).length
      lb : LT.lt 0 (Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1))
      ⊢ LE.le (List.count DyckStep.D (List.drop 1 (List.take (Min.min (HAdd.hAdd 1 i …
    -/
    have eq := hn.2 lb ub
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      l1 : Eq (List.take 1 ↑p) (List.cons ((↑p).head h) List.nil)
      l3 : Eq (HSub.hSub (↑p).length 1) (HAdd.hAdd (HSub.hSub (HSub.hSub (↑p).length …
      ub : LT.lt (Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1)) (↑p).length
      lb : LT.lt 0 (Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1))
      eq : LT.lt (List.count DyckStep.D (List.take (Min.min (HAdd.hAdd 1 i) (HSub.hS …
      ⊢ LE.le (List.count DyckStep.D (List.drop 1 (List.take (Min.min (HAdd.hAdd 1 i …
    -/
    set j := min (1 + i) (p.toList.length - 1)
    rw [← (p.toList.take j).take_append_drop 1, count_append, count_append, take_take,
      min_eq_left (by omega), l1, head_eq_U] at eq
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      l1 : Eq (List.take 1 ↑p) (List.cons ((↑p).head h) List.nil)
      l3 : Eq (HSub.hSub (↑p).length 1) (HAdd.hAdd (HSub.hSub (HSub.hSub (↑p).length …
      j : Nat := Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1)
      ub : LT.lt j (↑p).length
      lb : LT.lt 0 j
      eq : LT.lt (HAdd.hAdd (List.count DyckStep.D (List.cons DyckStep.U List.nil))  …
      ⊢ LE.le (List.count DyckStep.D (List.drop 1 (List.take j ↑p))) (List.count Dyc …
    -/
    simp only [count_singleton', ite_true, ite_false] at eq
    /-
      p q : DyckWord
      hn : p.IsNested
      i : Nat
      h : Ne (↑p) List.nil
      l1 : Eq (List.take 1 ↑p) (List.cons ((↑p).head h) List.nil)
      l3 : Eq (HSub.hSub (↑p).length 1) (HAdd.hAdd (HSub.hSub (HSub.hSub (↑p).length …
      j : Nat := Min.min (HAdd.hAdd 1 i) (HSub.hSub (↑p).length 1)
      ub : LT.lt j (↑p).length
      lb : LT.lt 0 j
      eq : LT.lt (HAdd.hAdd (ite (Eq DyckStep.U DyckStep.D) 1 0) (List.count DyckSte …
      ⊢ LE.le (List.count DyckStep.D (List.drop 1 (List.take j ↑p))) (List.count Dyc …
    -/
    omega
    /-
      🎉 no goals
    -/


variable (p) in
lemma nest_denest (hn) : (p.denest hn).nest = p := by
  /-
    p : DyckWord
    hn : p.IsNested
    ⊢ Eq (p.denest hn).nest p
  -/
  simpa [DyckWord.ext_iff] using p.cons_tail_dropLast_concat hn.1
  /-
    🎉 no goals
  -/


variable (p) in
lemma denest_nest : p.nest.denest .nest = p := by
  /-
    p : DyckWord
    ⊢ Eq (p.nest.denest ⋯) p
  -/
  simp_rw [nest, denest, DyckWord.ext_iff, dropLast_concat]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


variable (p) in
/-- The semilength of a Dyck word is half of the number of `DyckStep`s in it, or equivalently
its number of `U`s. -/
def semilength : ℕ := p.toList.count U


@[simp] lemma semilength_zero : semilength 0 = 0 := rfl

@[simp] lemma semilength_add : (p + q).semilength = p.semilength + q.semilength := count_append ..

                                                                           /-
                                                                             p : DyckWord
                                                                             ⊢ Eq p.nest.semilength (HAdd.hAdd p.semilength 1)
                                                                           -/
@[simp] lemma semilength_nest : p.nest.semilength = p.semilength + 1 := by simp [semilength, nest]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma semilength_eq_count_D : p.semilength = p.toList.count D := by
  /-
    p : DyckWord
    ⊢ Eq p.semilength (List.count DyckStep.D ↑p)
  -/
  rw [← count_U_eq_count_D]; rfl
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma two_mul_semilength_eq_length : 2 * p.semilength = p.toList.length := by
  /-
    p : DyckWord
    ⊢ Eq (HMul.hMul 2 p.semilength) (↑p).length
  -/
  nth_rw 1 [two_mul, semilength, p.count_U_eq_count_D, semilength]
  /-
    p : DyckWord
    ⊢ Eq (HAdd.hAdd (List.count DyckStep.D ↑p) (List.count DyckStep.U ↑p)) (↑p).le …
  -/
  convert (p.toList.length_eq_countP_add_countP (· == D)).symm
  /-
    case h.e'_2.h.e'_6.h.e
    p : DyckWord
    ⊢ Eq (List.count DyckStep.U) (List.countP fun a => Decidable.decide (Not (Eq ( …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  rw [count]; congr!; rename_i s; cases s <;> tauto
                                              /-
                                                🎉 no goals
                                              -/


variable (p) in
/-- `p.firstReturn` is 0 if `p = 0` and the index of the `D` matching the initial `U` otherwise. -/
def firstReturn : ℕ :=
  (range p.toList.length).findIdx fun i ↦
    (p.toList.take (i + 1)).count U = (p.toList.take (i + 1)).count D


@[simp] lemma firstReturn_zero : firstReturn 0 = 0 := rfl


include h in
lemma firstReturn_pos : 0 < p.firstReturn := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ LT.lt 0 p.firstReturn
  -/
  by_contra! f
  /-
    p : DyckWord
    h : Ne p 0
    f : LE.le p.firstReturn 0
    ⊢ False
  -/
  rw [Nat.le_zero, firstReturn, findIdx_eq] at f
  #adaptation_note
  /--
  If we don't swap, then the second goal is dropped after completing the first goal.
  What's going on?
  -/
  /-
    p : DyckWord
    h : Ne p 0
    f✝ : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (L …
    f : And (Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd …
    ⊢ False
  -/
  swap
    /-
      p : DyckWord
      h : Ne p 0
      f : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (Li …
      ⊢ LT.lt 0 (List.range (↑p).length).length
    -/
  · rw [length_range, length_pos]
    /-
      p : DyckWord
      h : Ne p 0
      f : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (Li …
      ⊢ Ne (↑p) List.nil
    -/
    exact toList_ne_nil.mpr h
    /-
      🎉 no goals
    -/
    /-
      p : DyckWord
      h : Ne p 0
      f✝ : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (L …
      f : And (Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd …
      ⊢ False
    -/
  · rw [getElem_range] at f
    /-
      p : DyckWord
      h : Ne p 0
      f✝ : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (L …
      f : And (Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd …
      ⊢ False
    -/
    simp at f
    /-
      p : DyckWord
      h : Ne p 0
      f✝ : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (L …
      f : Eq (List.count DyckStep.U (List.take 1 ↑p)) (List.count DyckStep.D (List.t …
      ⊢ False
    -/
    rw [← p.cons_tail_dropLast_concat h] at f
    /-
      p : DyckWord
      h : Ne p 0
      f✝ : Eq (List.findIdx (fun i => Decidable.decide (Eq (List.count DyckStep.U (L …
      f : Eq (List.count DyckStep.U (List.take 1 (HAppend.hAppend (List.cons DyckSte …
      ⊢ False
    -/
    simp at f
    /-
      🎉 no goals
    -/


include h in
lemma firstReturn_lt_length : p.firstReturn < p.toList.length := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ LT.lt p.firstReturn (↑p).length
  -/
  have lp := length_pos_of_ne_nil (toList_ne_nil.mpr h)
  /-
    p : DyckWord
    h : Ne p 0
    lp : LT.lt 0 (↑p).length
    ⊢ LT.lt p.firstReturn (↑p).length
  -/
  rw [← length_range p.toList.length]
  /-
    p : DyckWord
    h : Ne p 0
    lp : LT.lt 0 (↑p).length
    ⊢ LT.lt p.firstReturn (List.range (↑p).length).length
  -/
  apply findIdx_lt_length_of_exists
  /-
    case h
    p : DyckWord
    h : Ne p 0
    lp : LT.lt 0 (↑p).length
    ⊢ Exists fun x => And (Membership.mem (List.range (↑p).length) x) (Eq (Decidab …
  -/
  simp only [mem_range, decide_eq_true_eq]
  /-
    case h
    p : DyckWord
    h : Ne p 0
    lp : LT.lt 0 (↑p).length
    ⊢ Exists fun x => And (LT.lt x (↑p).length) (Eq (List.count DyckStep.U (List.t …
  -/
  use p.toList.length - 1
  exact ⟨by omega, by rw [Nat.sub_add_cancel lp, take_of_length_le (le_refl _),
    p.count_U_eq_count_D]⟩


include h in
lemma count_take_firstReturn_add_one :
    (p.toList.take (p.firstReturn + 1)).count U = (p.toList.take (p.firstReturn + 1)).count D := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ Eq (List.count DyckStep.U (List.take (HAdd.hAdd p.firstReturn 1) ↑p)) (List. …
  -/
  have := findIdx_getElem (w := (length_range p.toList.length).symm ▸ firstReturn_lt_length h)
  /-
    p : DyckWord
    h : Ne p 0
    this : Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd ( …
    ⊢ Eq (List.count DyckStep.U (List.take (HAdd.hAdd p.firstReturn 1) ↑p)) (List. …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


lemma count_D_lt_count_U_of_lt_firstReturn {i : ℕ} (hi : i < p.firstReturn) :
    (p.toList.take (i + 1)).count D < (p.toList.take (i + 1)).count U := by
  /-
    p : DyckWord
    i : Nat
    hi : LT.lt i p.firstReturn
    ⊢ LT.lt (List.count DyckStep.D (List.take (HAdd.hAdd i 1) ↑p)) (List.count Dyc …
  -/
  have ne := not_of_lt_findIdx hi
  /-
    p : DyckWord
    i : Nat
    hi : LT.lt i p.firstReturn
    ne : Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd (Ge …
    ⊢ LT.lt (List.count DyckStep.D (List.take (HAdd.hAdd i 1) ↑p)) (List.count Dyc …
  -/
  rw [decide_eq_false_iff_not, ← ne_eq, getElem_range] at ne
  /-
    p : DyckWord
    i : Nat
    hi : LT.lt i p.firstReturn
    ne : Ne (List.count DyckStep.U (List.take (HAdd.hAdd i 1) ↑p)) (List.count Dyc …
    ⊢ LT.lt (List.count DyckStep.D (List.take (HAdd.hAdd i 1) ↑p)) (List.count Dyc …
  -/
  exact lt_of_le_of_ne (p.count_D_le_count_U (i + 1)) ne.symm
  /-
    🎉 no goals
  -/


@[simp]
lemma firstReturn_add : (p + q).firstReturn = if p = 0 then q.firstReturn else p.firstReturn := by
  /-
    p q : DyckWord
    ⊢ Eq (HAdd.hAdd p q).firstReturn (ite (Eq p 0) q.firstReturn p.firstReturn)
  -/
  split_ifs with h; · simp [h]
                      /-
                        🎉 no goals
                      -/
  /-
    case neg
    p q : DyckWord
    h : Not (Eq p 0)
    ⊢ Eq (HAdd.hAdd p q).firstReturn p.firstReturn
  -/
  have u : (p + q).toList = p.toList ++ q.toList := rfl
  /-
    case neg
    p q : DyckWord
    h : Not (Eq p 0)
    u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
    ⊢ Eq (HAdd.hAdd p q).firstReturn p.firstReturn
  -/
  rw [firstReturn, findIdx_eq]
    /-
      case neg
      p q : DyckWord
      h : Not (Eq p 0)
      u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
      ⊢ And (Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd ( …
    -/
  · simp_rw [u, decide_eq_true_eq, getElem_range]
    /-
      case neg
      p q : DyckWord
      h : Not (Eq p 0)
      u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
      ⊢ And (Eq (List.count DyckStep.U (List.take (HAdd.hAdd p.firstReturn 1) (HAppe …
    -/
    have v := firstReturn_lt_length h
    /-
      case neg
      p q : DyckWord
      h : Not (Eq p 0)
      u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
      v : LT.lt p.firstReturn (↑p).length
      ⊢ And (Eq (List.count DyckStep.U (List.take (HAdd.hAdd p.firstReturn 1) (HAppe …
    -/
    constructor
    · rw [take_append_eq_append_take, show p.firstReturn + 1 - p.toList.length = 0 by omega,
        take_zero, append_nil, count_take_firstReturn_add_one h]
      /-
        case neg.right
        p q : DyckWord
        h : Not (Eq p 0)
        u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
        v : LT.lt p.firstReturn (↑p).length
        ⊢ ∀ (j : Nat), LT.lt j p.firstReturn → Eq (Decidable.decide (Eq (List.count Dy …
      -/
    · intro j hj
      rw [take_append_eq_append_take, show j + 1 - p.toList.length = 0 by omega,
        take_zero, append_nil]
      /-
        case neg.right
        p q : DyckWord
        h : Not (Eq p 0)
        u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
        v : LT.lt p.firstReturn (↑p).length
        j : Nat
        hj : LT.lt j p.firstReturn
        ⊢ Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd j 1) ↑ …
      -/
      simpa using (count_D_lt_count_U_of_lt_firstReturn hj).ne'
      /-
        🎉 no goals
      -/
    /-
      case neg
      p q : DyckWord
      h : Not (Eq p 0)
      u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
      ⊢ LT.lt p.firstReturn (List.range (↑(HAdd.hAdd p q)).length).length
    -/
  · rw [length_range, u, length_append]
    /-
      case neg
      p q : DyckWord
      h : Not (Eq p 0)
      u : Eq (↑(HAdd.hAdd p q)) (HAppend.hAppend ↑p ↑q)
      ⊢ LT.lt p.firstReturn (HAdd.hAdd (↑p).length (↑q).length)
    -/
    exact Nat.lt_add_right _ (firstReturn_lt_length h)
    /-
      🎉 no goals
    -/


@[simp]
lemma firstReturn_nest : p.nest.firstReturn = p.toList.length + 1 := by
  /-
    p : DyckWord
    ⊢ Eq p.nest.firstReturn (HAdd.hAdd (↑p).length 1)
  -/
  have u : p.nest.toList = U :: p.toList ++ [D] := rfl
  /-
    p : DyckWord
    u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
    ⊢ Eq p.nest.firstReturn (HAdd.hAdd (↑p).length 1)
  -/
  rw [firstReturn, findIdx_eq]
    /-
      p : DyckWord
      u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
      ⊢ And (Eq (Decidable.decide (Eq (List.count DyckStep.U (List.take (HAdd.hAdd ( …
    -/
  · simp_rw [u, decide_eq_true_eq, getElem_range]
    /-
      p : DyckWord
      u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
      ⊢ And (Eq (List.count DyckStep.U (List.take (HAdd.hAdd (HAdd.hAdd (↑p).length  …
    -/
    constructor
      /-
        case left
        p : DyckWord
        u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
        ⊢ Eq (List.count DyckStep.U (List.take (HAdd.hAdd (HAdd.hAdd (↑p).length 1) 1) …
      -/
    · rw [take_of_length_le (by simp), ← u, p.nest.count_U_eq_count_D]
      /-
        🎉 no goals
      -/
      /-
        case right
        p : DyckWord
        u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
        ⊢ ∀ (j : Nat), LT.lt j (HAdd.hAdd (↑p).length 1) → Eq (Decidable.decide (Eq (L …
      -/
    · intro j hj
      simp_rw [cons_append, take_succ_cons, count_cons, beq_self_eq_true, ite_true,
        beq_iff_eq, reduceCtorEq, ite_false, take_append_eq_append_take,
        show j - p.toList.length = 0 by omega, take_zero, append_nil]
      /-
        case right
        p : DyckWord
        u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
        j : Nat
        hj : LT.lt j (HAdd.hAdd (↑p).length 1)
        ⊢ Eq (Decidable.decide (Eq (HAdd.hAdd (List.count DyckStep.U (List.take j ↑p)) …
      -/
      have := p.count_D_le_count_U j
      /-
        case right
        p : DyckWord
        u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
        j : Nat
        hj : LT.lt j (HAdd.hAdd (↑p).length 1)
        this : LE.le (List.count DyckStep.D (List.take j ↑p)) (List.count DyckStep.U ( …
        ⊢ Eq (Decidable.decide (Eq (HAdd.hAdd (List.count DyckStep.U (List.take j ↑p)) …
      -/
      simp only [add_zero, decide_eq_false_iff_not, ne_eq]
      /-
        case right
        p : DyckWord
        u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
        j : Nat
        hj : LT.lt j (HAdd.hAdd (↑p).length 1)
        this : LE.le (List.count DyckStep.D (List.take j ↑p)) (List.count DyckStep.U ( …
        ⊢ Not (Eq (HAdd.hAdd (List.count DyckStep.U (List.take j ↑p)) 1) (List.count D …
      -/
      omega
      /-
        🎉 no goals
      -/
    /-
      p : DyckWord
      u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
      ⊢ LT.lt (HAdd.hAdd (↑p).length 1) (List.range (↑p.nest).length).length
    -/
  · simp_rw [length_range, u, length_append, length_cons]
    /-
      p : DyckWord
      u : Eq (↑p.nest) (HAppend.hAppend (List.cons DyckStep.U ↑p) (List.cons DyckSte …
      ⊢ LT.lt (HAdd.hAdd (↑p).length 1) (HAdd.hAdd (HAdd.hAdd (↑p).length 1) (HAdd.h …
    -/
    exact Nat.lt_add_one _
    /-
      🎉 no goals
    -/


variable (p) in
/-- The left part of the Dyck word decomposition,
inside the `U, D` pair that `firstReturn` refers to. `insidePart 0 = 0`. -/
def insidePart : DyckWord :=
  if h : p = 0 then 0 else
  (p.take (p.firstReturn + 1) (count_take_firstReturn_add_one h)).denest
        /-
          p q : DyckWord
          h✝ : Ne p 0
          h : Not (Eq p 0)
          ⊢ Ne (p.take (HAdd.hAdd p.firstReturn 1) ⋯) 0
        -/
    ⟨by rw [← toList_ne_nil, take]; simpa using toList_ne_nil.mpr h, fun i lb ub ↦ by
                                    /-
                                      🎉 no goals
                                    -/
      /-
        p q : DyckWord
        h✝ : Ne p 0
        h : Not (Eq p 0)
        i : Nat
        lb : LT.lt 0 i
        ub : LT.lt i (↑(p.take (HAdd.hAdd p.firstReturn 1) ⋯)).length
        ⊢ LT.lt (List.count DyckStep.D (List.take i ↑(p.take (HAdd.hAdd p.firstReturn  …
      -/
      simp only [take, length_take, lt_min_iff] at ub ⊢
      /-
        p q : DyckWord
        h✝ : Ne p 0
        h : Not (Eq p 0)
        i : Nat
        lb : LT.lt 0 i
        ub : And (LT.lt i (HAdd.hAdd p.firstReturn 1)) (LT.lt i (↑p).length)
        ⊢ LT.lt (List.count DyckStep.D (List.take i (List.take (HAdd.hAdd p.firstRetur …
      -/
      replace ub := ub.1
      /-
        p q : DyckWord
        h✝ : Ne p 0
        h : Not (Eq p 0)
        i : Nat
        lb : LT.lt 0 i
        ub : LT.lt i (HAdd.hAdd p.firstReturn 1)
        ⊢ LT.lt (List.count DyckStep.D (List.take i (List.take (HAdd.hAdd p.firstRetur …
      -/
      rw [take_take, min_eq_left ub.le]
      /-
        p q : DyckWord
        h✝ : Ne p 0
        h : Not (Eq p 0)
        i : Nat
        lb : LT.lt 0 i
        ub : LT.lt i (HAdd.hAdd p.firstReturn 1)
        ⊢ LT.lt (List.count DyckStep.D (List.take i ↑p)) (List.count DyckStep.U (List. …
      -/
      rw [show i = i - 1 + 1 by omega] at ub ⊢
      /-
        p q : DyckWord
        h✝ : Ne p 0
        h : Not (Eq p 0)
        i : Nat
        lb : LT.lt 0 i
        ub : LT.lt (HAdd.hAdd (HSub.hSub i 1) 1) (HAdd.hAdd p.firstReturn 1)
        ⊢ LT.lt (List.count DyckStep.D (List.take (HAdd.hAdd (HSub.hSub i 1) 1) ↑p)) ( …
      -/
      rw [Nat.add_lt_add_iff_right] at ub
      /-
        p q : DyckWord
        h✝ : Ne p 0
        h : Not (Eq p 0)
        i : Nat
        lb : LT.lt 0 i
        ub : LT.lt (HSub.hSub i 1) p.firstReturn
        ⊢ LT.lt (List.count DyckStep.D (List.take (HAdd.hAdd (HSub.hSub i 1) 1) ↑p)) ( …
      -/
      exact count_D_lt_count_U_of_lt_firstReturn ub⟩
      /-
        🎉 no goals
      -/


variable (p) in
/-- The right part of the Dyck word decomposition,
outside the `U, D` pair that `firstReturn` refers to. `outsidePart 0 = 0`. -/
def outsidePart : DyckWord :=
  if h : p = 0 then 0 else p.drop (p.firstReturn + 1) (count_take_firstReturn_add_one h)


                                                       /-
                                                         ⊢ Eq (DyckWord.insidePart 0) 0
                                                       -/
@[simp] lemma insidePart_zero : insidePart 0 = 0 := by simp [insidePart]
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                         /-
                                                           ⊢ Eq (DyckWord.outsidePart 0) 0
                                                         -/
@[simp] lemma outsidePart_zero : outsidePart 0 = 0 := by simp [outsidePart]
                                                         /-
                                                           🎉 no goals
                                                         -/


include h in
@[simp]
lemma insidePart_add : (p + q).insidePart = p.insidePart := by
  simp_rw [insidePart, firstReturn_add, add_eq_zero', h, false_and, dite_false, ite_false,
    DyckWord.ext_iff, take]
  /-
    p q : DyckWord
    h : Ne p 0
    ⊢ Eq ↑({ toList := List.take (HAdd.hAdd p.firstReturn 1) ↑(HAdd.hAdd p q), cou …
  -/
  congr 3
  /-
    case e_self.e_p.e_toList
    p q : DyckWord
    h : Ne p 0
    ⊢ Eq (List.take (HAdd.hAdd p.firstReturn 1) ↑(HAdd.hAdd p q)) (List.take (HAdd …
  -/
  exact take_append_of_le_length (firstReturn_lt_length h)
  /-
    🎉 no goals
  -/


include h in
@[simp]
lemma outsidePart_add : (p + q).outsidePart = p.outsidePart + q := by
  simp_rw [outsidePart, firstReturn_add, add_eq_zero', h, false_and, dite_false, ite_false,
    DyckWord.ext_iff, drop]
  /-
    p q : DyckWord
    h : Ne p 0
    ⊢ Eq (List.drop (HAdd.hAdd p.firstReturn 1) ↑(HAdd.hAdd p q)) ↑(HAdd.hAdd { to …
  -/
  exact drop_append_of_le_length (firstReturn_lt_length h)
  /-
    🎉 no goals
  -/


@[simp]
lemma insidePart_nest : p.nest.insidePart = p := by
  /-
    p : DyckWord
    ⊢ Eq p.nest.insidePart p
  -/
  simp_rw [insidePart, nest_ne_zero, dite_false, firstReturn_nest]
  /-
    p : DyckWord
    ⊢ Eq ((p.nest.take (HAdd.hAdd (HAdd.hAdd (↑p).length 1) 1) ⋯).denest ⋯) p
  -/
  convert p.denest_nest; rw [DyckWord.ext_iff]; apply take_of_length_le
  /-
    case h.e'_2.h.e'_1.h
    p : DyckWord
    ⊢ LE.le (↑p.nest).length (HAdd.hAdd (HAdd.hAdd (↑p).length 1) 1)
  -/
  simp_rw [nest, length_append, length_singleton]; omega
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
lemma outsidePart_nest : p.nest.outsidePart = 0 := by
  /-
    p : DyckWord
    ⊢ Eq p.nest.outsidePart 0
  -/
  simp_rw [outsidePart, nest_ne_zero, dite_false, firstReturn_nest]
  /-
    p : DyckWord
    ⊢ Eq (p.nest.drop (HAdd.hAdd (HAdd.hAdd (↑p).length 1) 1) ⋯) 0
  -/
  rw [DyckWord.ext_iff]; apply drop_of_length_le
  /-
    case h
    p : DyckWord
    ⊢ LE.le (↑p.nest).length (HAdd.hAdd (HAdd.hAdd (↑p).length 1) 1)
  -/
  simp_rw [nest, length_append, length_singleton]; omega
                                                   /-
                                                     🎉 no goals
                                                   -/


include h in
@[simp]
theorem nest_insidePart_add_outsidePart : p.insidePart.nest + p.outsidePart = p := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ Eq (HAdd.hAdd p.insidePart.nest p.outsidePart) p
  -/
  simp_rw [insidePart, outsidePart, h, dite_false, nest_denest, DyckWord.ext_iff]
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ Eq ↑(HAdd.hAdd (p.take (HAdd.hAdd p.firstReturn 1) ⋯) (p.drop (HAdd.hAdd p.f …
  -/
  apply take_append_drop
  /-
    🎉 no goals
  -/


include h in
lemma semilength_insidePart_add_semilength_outsidePart_add_one :
    p.insidePart.semilength + p.outsidePart.semilength + 1 = p.semilength := by
  rw [← congrArg semilength (nest_insidePart_add_outsidePart h), semilength_add, semilength_nest,
    add_right_comm]


include h in
theorem semilength_insidePart_lt : p.insidePart.semilength < p.semilength := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ LT.lt p.insidePart.semilength p.semilength
  -/
  have := semilength_insidePart_add_semilength_outsidePart_add_one h
  /-
    p : DyckWord
    h : Ne p 0
    this : Eq (HAdd.hAdd (HAdd.hAdd p.insidePart.semilength p.outsidePart.semileng …
    ⊢ LT.lt p.insidePart.semilength p.semilength
  -/
  omega
  /-
    🎉 no goals
  -/


include h in
theorem semilength_outsidePart_lt : p.outsidePart.semilength < p.semilength := by
  /-
    p : DyckWord
    h : Ne p 0
    ⊢ LT.lt p.outsidePart.semilength p.semilength
  -/
  have := semilength_insidePart_add_semilength_outsidePart_add_one h
  /-
    p : DyckWord
    h : Ne p 0
    this : Eq (HAdd.hAdd (HAdd.hAdd p.insidePart.semilength p.outsidePart.semileng …
    ⊢ LT.lt p.outsidePart.semilength p.semilength
  -/
  omega
  /-
    🎉 no goals
  -/


instance : Preorder DyckWord where
  le := Relation.ReflTransGen (fun p q ↦ p = q.insidePart ∨ p = q.outsidePart)
  le_refl _ := Relation.ReflTransGen.refl
  le_trans _ _ _ := Relation.ReflTransGen.trans


lemma le_add_self (p q : DyckWord) : q ≤ p + q := by
  /-
    p q : DyckWord
    ⊢ LE.le q (HAdd.hAdd p q)
  -/
  by_cases h : p = 0
    /-
      case pos
      p q : DyckWord
      h : Eq p 0
      ⊢ LE.le q (HAdd.hAdd p q)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p q : DyckWord
      h : Not (Eq p 0)
      ⊢ LE.le q (HAdd.hAdd p q)
    -/
  · have := semilength_outsidePart_lt h
    exact (le_add_self p.outsidePart q).trans
      (Relation.ReflTransGen.single (Or.inr (outsidePart_add h).symm))
termination_by p.semilength


variable (p) in protected lemma zero_le : 0 ≤ p := add_zero p ▸ le_add_self p 0


lemma infix_of_le (h : p ≤ q) : p.toList <:+: q.toList := by
  induction h with
  | refl => exact infix_refl _
  | tail _pm mq ih =>
    rename_i m r
    rcases eq_or_ne r 0 with rfl | hr
    · rw [insidePart_zero, outsidePart_zero, or_self] at mq
      rwa [mq] at ih
    · have : [U] ++ r.insidePart ++ [D] ++ r.outsidePart = r :=
        DyckWord.ext_iff.mp (nest_insidePart_add_outsidePart hr)
      rcases mq with hm | hm
      · have : r.insidePart <:+: r.toList := by
          use [U], [D] ++ r.outsidePart; rwa [← append_assoc]
        exact ih.trans (hm ▸ this)
      · have : r.outsidePart <:+: r.toList := by
          use [U] ++ r.insidePart ++ [D], []; rwa [append_nil]
        exact ih.trans (hm ▸ this)


lemma le_of_suffix (h : p.toList <:+ q.toList) : p ≤ q := by
  /-
    p q : DyckWord
    h : (↑p).IsSuffix ↑q
    ⊢ LE.le p q
  -/
  obtain ⟨r', h⟩ := h
  have hc : (q.toList.take (q.toList.length - p.toList.length)).count U =
      (q.toList.take (q.toList.length - p.toList.length)).count D := by
    have hq := q.count_U_eq_count_D
    rw [← h] at hq ⊢
    rw [count_append, count_append, p.count_U_eq_count_D, Nat.add_right_cancel_iff] at hq
    simp [hq]
  /-
    case intro
    p q : DyckWord
    r' : List DyckStep
    h : Eq (HAppend.hAppend r' ↑p) ↑q
    hc : Eq (List.count DyckStep.U (List.take (HSub.hSub (↑q).length (↑p).length)  …
    ⊢ LE.le p q
  -/
  let r : DyckWord := q.take _ hc
  have e : r' = r := by
    simp_rw [r, take, ← h, length_append, add_tsub_cancel_right, take_left']
  /-
    case intro
    p q : DyckWord
    r' : List DyckStep
    h : Eq (HAppend.hAppend r' ↑p) ↑q
    hc : Eq (List.count DyckStep.U (List.take (HSub.hSub (↑q).length (↑p).length)  …
    r : DyckWord := q.take (HSub.hSub (↑q).length (↑p).length) hc
    e : Eq r' ↑r
    ⊢ LE.le p q
  -/
  rw [e] at h; replace h : r + p = q := DyckWord.ext h; rw [← h]; exact le_add_self ..
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Partial order on Dyck words: `p ≤ q` if a (possibly empty) sequence of
`insidePart` and `outsidePart` operations can turn `q` into `p`. -/
instance : PartialOrder DyckWord where
  le_antisymm p q pq qp := by
    /-
      p✝ q✝ : DyckWord
      h : Ne p✝ 0
      p q : DyckWord
      pq : LE.le p q
      qp : LE.le q p
      ⊢ Eq p q
    -/
    have h₁ := infix_of_le pq
    /-
      p✝ q✝ : DyckWord
      h : Ne p✝ 0
      p q : DyckWord
      pq : LE.le p q
      qp : LE.le q p
      h₁ : (↑p).IsInfix ↑q
      ⊢ Eq p q
    -/
    have h₂ := infix_of_le qp
    /-
      p✝ q✝ : DyckWord
      h : Ne p✝ 0
      p q : DyckWord
      pq : LE.le p q
      qp : LE.le q p
      h₁ : (↑p).IsInfix ↑q
      h₂ : (↑q).IsInfix ↑p
      ⊢ Eq p q
    -/
    exact DyckWord.ext <| h₁.eq_of_length <| h₁.length_le.antisymm h₂.length_le
    /-
      🎉 no goals
    -/


protected lemma pos_iff_ne_zero : 0 < p ↔ p ≠ 0 := by
  /-
    p : DyckWord
    ⊢ Iff (LT.lt 0 p) (Ne p 0)
  -/
  rw [ne_comm, iff_comm, ne_iff_lt_iff_le]
  /-
    p : DyckWord
    ⊢ LE.le 0 p
  -/
  exact DyckWord.zero_le p
  /-
    🎉 no goals
  -/


lemma monotone_semilength : Monotone semilength := fun p q pq ↦ by
  induction pq with
  | refl => rfl
  | tail _ mq ih =>
    rename_i m r _
    rcases eq_or_ne r 0 with rfl | hr
    · rw [insidePart_zero, outsidePart_zero, or_self] at mq
      rwa [mq] at ih
    · rcases mq with hm | hm
      · exact ih.trans (hm ▸ semilength_insidePart_lt hr).le
      · exact ih.trans (hm ▸ semilength_outsidePart_lt hr).le


lemma strictMono_semilength : StrictMono semilength := fun p q pq ↦ by
  /-
    p q : DyckWord
    pq : LT.lt p q
    ⊢ LT.lt p.semilength q.semilength
  -/
  obtain ⟨plq, pnq⟩ := lt_iff_le_and_ne.mp pq
  /-
    case intro
    p q : DyckWord
    pq : LT.lt p q
    plq : LE.le p q
    pnq : Ne p q
    ⊢ LT.lt p.semilength q.semilength
  -/
  apply lt_of_le_of_ne (monotone_semilength plq)
  /-
    case intro
    p q : DyckWord
    pq : LT.lt p q
    plq : LE.le p q
    pnq : Ne p q
    ⊢ Ne p.semilength q.semilength
  -/
  contrapose! pnq
  /-
    case intro
    p q : DyckWord
    pq : LT.lt p q
    plq : LE.le p q
    pnq : Eq p.semilength q.semilength
    ⊢ Eq p q
  -/
  replace pnq := congr(2 * $(pnq))
  /-
    case intro
    p q : DyckWord
    pq : LT.lt p q
    plq : LE.le p q
    pnq : Eq (HMul.hMul 2 p.semilength) (HMul.hMul 2 q.semilength)
    ⊢ Eq p q
  -/
  simp_rw [two_mul_semilength_eq_length] at pnq
  /-
    case intro
    p q : DyckWord
    pq : LT.lt p q
    plq : LE.le p q
    pnq : Eq (↑p).length (↑q).length
    ⊢ Eq p q
  -/
  exact DyckWord.ext ((infix_of_le plq).eq_of_length pnq)
  /-
    🎉 no goals
  -/


/-- Convert a Dyck word to a binary rooted tree.

`f(0) = nil`. For a nonzero word find the `D` that matches the initial `U`,
which has index `p.firstReturn`, then let `x` be everything strictly between said `U` and `D`,
and `y` be everything strictly after said `D`. `p = x.nest + y` with `x, y` (possibly empty)
Dyck words. `f(p) = f(x) △ f(y)`, where △ (defined in `Mathlib.Data.Tree`) joins two subtrees
to a new root node. -/
private def equivTreeToFun (p : DyckWord) : Tree Unit :=
  if h : p = 0 then nil else
    have := semilength_insidePart_lt h
    have := semilength_outsidePart_lt h
    equivTreeToFun p.insidePart △ equivTreeToFun p.outsidePart
termination_by p.semilength


/-- Convert a binary rooted tree to a Dyck word.

`g(nil) = 0`. A nonempty tree with left subtree `l` and right subtree `r`
is sent to `g(l).nest + g(r)`. -/
private def equivTreeInvFun : Tree Unit → DyckWord
  | Tree.nil => 0
  | Tree.node _ l r => (equivTreeInvFun l).nest + equivTreeInvFun r


@[nolint unusedHavesSuffices]
private lemma equivTree_left_inv (p) : equivTreeInvFun (equivTreeToFun p) = p := by
  /-
    p : DyckWord
    ⊢ Eq (DyckWord.equivTreeInvFun (DyckWord.equivTreeToFun p)) p
  -/
  by_cases h : p = 0
    /-
      case pos
      p : DyckWord
      h : Eq p 0
      ⊢ Eq (DyckWord.equivTreeInvFun (DyckWord.equivTreeToFun p)) p
    -/
  · simp [h, equivTreeToFun, equivTreeInvFun]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      ⊢ Eq (DyckWord.equivTreeInvFun (DyckWord.equivTreeToFun p)) p
    -/
  · rw [equivTreeToFun]
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      ⊢ Eq (DyckWord.equivTreeInvFun (dite (Eq p 0) (fun h => Tree.nil) fun h => let …
    -/
    simp_rw [h, dite_false, equivTreeInvFun]
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      ⊢ Eq (HAdd.hAdd (DyckWord.equivTreeInvFun (DyckWord.equivTreeToFun p.insidePar …
    -/
    have := semilength_insidePart_lt h
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      this : LT.lt p.insidePart.semilength p.semilength
      ⊢ Eq (HAdd.hAdd (DyckWord.equivTreeInvFun (DyckWord.equivTreeToFun p.insidePar …
    -/
    have := semilength_outsidePart_lt h
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      this✝ : LT.lt p.insidePart.semilength p.semilength
      this : LT.lt p.outsidePart.semilength p.semilength
      ⊢ Eq (HAdd.hAdd (DyckWord.equivTreeInvFun (DyckWord.equivTreeToFun p.insidePar …
    -/
    rw [equivTree_left_inv p.insidePart, equivTree_left_inv p.outsidePart]
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      this✝ : LT.lt p.insidePart.semilength p.semilength
      this : LT.lt p.outsidePart.semilength p.semilength
      ⊢ Eq (HAdd.hAdd p.insidePart.nest p.outsidePart) p
    -/
    exact nest_insidePart_add_outsidePart h
    /-
      🎉 no goals
    -/
termination_by p.semilength


@[nolint unusedHavesSuffices]
private lemma equivTree_right_inv : ∀ t, equivTreeToFun (equivTreeInvFun t) = t
                   /-
                     ⊢ Eq (DyckWord.equivTreeToFun (DyckWord.equivTreeInvFun Tree.nil)) Tree.nil
                   -/
  | Tree.nil => by simp [equivTreeInvFun, equivTreeToFun]
                   /-
                     🎉 no goals
                   -/
                          /-
                            a✝² : Unit
                            a✝¹ a✝ : Tree Unit
                            ⊢ Eq (DyckWord.equivTreeToFun (DyckWord.equivTreeInvFun (Tree.node a✝² a✝¹ a✝) …
                          -/
  | Tree.node _ _ _ => by simp [equivTreeInvFun, equivTreeToFun, equivTree_right_inv]
                          /-
                            🎉 no goals
                          -/


/-- Equivalence between Dyck words and rooted binary trees. -/
def equivTree : DyckWord ≃ Tree Unit where
  toFun := equivTreeToFun
  invFun := equivTreeInvFun
  left_inv := equivTree_left_inv
  right_inv := equivTree_right_inv


@[nolint unusedHavesSuffices]
lemma semilength_eq_numNodes_equivTree (p) : p.semilength = (equivTree p).numNodes := by
  /-
    p : DyckWord
    ⊢ Eq p.semilength (DyckWord.equivTree p).numNodes
  -/
  by_cases h : p = 0
    /-
      case pos
      p : DyckWord
      h : Eq p 0
      ⊢ Eq p.semilength (DyckWord.equivTree p).numNodes
    -/
  · simp [h, equivTree, equivTreeToFun]
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      ⊢ Eq p.semilength (DyckWord.equivTree p).numNodes
    -/
  · rw [equivTree, Equiv.coe_fn_mk, equivTreeToFun]
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      ⊢ Eq p.semilength (dite (Eq p 0) (fun h => Tree.nil) fun h => letFun ⋯ fun thi …
    -/
    simp_rw [h, dite_false, numNodes]
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      ⊢ Eq p.semilength (HAdd.hAdd (HAdd.hAdd (DyckWord.equivTreeToFun p.insidePart) …
    -/
    have := semilength_insidePart_lt h
    /-
      case neg
      p : DyckWord
      h : Not (Eq p 0)
      this : LT.lt p.insidePart.semilength p.semilength
      ⊢ Eq p.semilength (HAdd.hAdd (HAdd.hAdd (DyckWord.equivTreeToFun p.insidePart) …
    -/
    have := semilength_outsidePart_lt h
    rw [← semilength_insidePart_add_semilength_outsidePart_add_one h,
      semilength_eq_numNodes_equivTree p.insidePart,
                                                       /-
                                                         case neg
                                                         p : DyckWord
                                                         h : Not (Eq p 0)
                                                         this✝ : LT.lt p.insidePart.semilength p.semilength
                                                         this : LT.lt p.outsidePart.semilength p.semilength
                                                         ⊢ Eq (HAdd.hAdd (HAdd.hAdd (DyckWord.equivTree p.insidePart).numNodes (DyckWor …
                                                       -/
      semilength_eq_numNodes_equivTree p.outsidePart]; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/
termination_by p.semilength


/-- Equivalence between Dyck words of semilength `n` and rooted binary trees with
`n` internal nodes. -/
def equivTreesOfNumNodesEq (n : ℕ) :
    { p : DyckWord // p.semilength = n } ≃ treesOfNumNodesEq n where
  toFun := fun ⟨p, _⟩ ↦ ⟨equivTree p, by
    /-
      p✝ q : DyckWord
      h : Ne p✝ 0
      n : Nat
      x✝ : Subtype fun p => Eq p.semilength n
      p : DyckWord
      property✝ : Eq p.semilength n
      ⊢ Membership.mem (Tree.treesOfNumNodesEq n) (DyckWord.equivTree p)
    -/
    rwa [mem_treesOfNumNodesEq, ← semilength_eq_numNodes_equivTree]⟩
    /-
      🎉 no goals
    -/
  invFun := fun ⟨tr, _⟩ ↦ ⟨equivTree.symm tr, by
    /-
      p q : DyckWord
      h : Ne p 0
      n : Nat
      x✝ : Subtype fun x => Membership.mem (Tree.treesOfNumNodesEq n) x
      tr : Tree Unit
      property✝ : Membership.mem (Tree.treesOfNumNodesEq n) tr
      ⊢ Eq (DyckWord.equivTree.symm tr).semilength n
    -/
    rwa [semilength_eq_numNodes_equivTree, ← mem_treesOfNumNodesEq, Equiv.apply_symm_apply]⟩
    /-
      🎉 no goals
    -/
                   /-
                     p q : DyckWord
                     h : Ne p 0
                     n : Nat
                     x✝ : Subtype fun p => Eq p.semilength n
                     ⊢ Eq ((fun x => DyckWord.equivTreesOfNumNodesEq.match_2 n (fun x => Subtype fu …
                   -/
  left_inv _ := by simp only [Equiv.symm_apply_apply]
                   /-
                     🎉 no goals
                   -/
                    /-
                      p q : DyckWord
                      h : Ne p 0
                      n : Nat
                      x✝ : Subtype fun x => Membership.mem (Tree.treesOfNumNodesEq n) x
                      ⊢ Eq ((fun x => DyckWord.equivTreesOfNumNodesEq.match_1 n (fun x => Subtype fu …
                    -/
  right_inv _ := by simp only [Equiv.apply_symm_apply]
                    /-
                      🎉 no goals
                    -/


instance {n : ℕ} : Fintype { p : DyckWord // p.semilength = n } :=
  Fintype.ofEquiv _ (equivTreesOfNumNodesEq n).symm


/-- There are `catalan n` Dyck words of semilength `n` (or length `2 * n`). -/
theorem card_dyckWord_semilength_eq_catalan (n : ℕ) :
    Fintype.card { p : DyckWord // p.semilength = n } = catalan n := by
  /-
    n : Nat
    ⊢ Eq (Fintype.card (Subtype fun p => Eq p.semilength n)) (catalan n)
  -/
  rw [← Fintype.ofEquiv_card (equivTreesOfNumNodesEq n), ← treesOfNumNodesEq_card_eq_catalan]
  /-
    n : Nat
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (Tree.treesOfNumNodesEq n) …
  -/
  convert Fintype.card_coe _
  /-
    🎉 no goals
  -/


/-- Extension for the `positivity` tactic: `p.firstReturn` is positive if `p` is nonzero. -/
@[positivity DyckWord.firstReturn _]
def evalDyckWordFirstReturn : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℕ), ~q(DyckWord.firstReturn $a) =>
    let ra ← core q(inferInstance) q(inferInstance) a
    assertInstancesCommute
    match ra with
    | .positive pa => pure (.positive q(DyckWord.firstReturn_pos ($pa).ne'))
    | .nonzero pa => pure (.positive q(DyckWord.firstReturn_pos $pa))
    | _ => pure .none
  | _, _, _ => throwError "not DyckWord.firstReturn"


