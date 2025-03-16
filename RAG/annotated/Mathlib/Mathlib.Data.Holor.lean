/-- `HolorIndex ds` is the type of valid index tuples used to identify an entry of a holor
of dimensions `ds`. -/
def HolorIndex (ds : List ℕ) : Type :=
  { is : List ℕ // Forall₂ (· < ·) is ds }


def take : ∀ {ds₁ : List ℕ}, HolorIndex (ds₁ ++ ds₂) → HolorIndex ds₁
  | ds, is => ⟨List.take (length ds) is.1, forall₂_take_append is.1 ds ds₂ is.2⟩


def drop : ∀ {ds₁ : List ℕ}, HolorIndex (ds₁ ++ ds₂) → HolorIndex ds₂
  | ds, is => ⟨List.drop (length ds) is.1, forall₂_drop_append is.1 ds ds₂ is.2⟩


theorem cast_type (is : List ℕ) (eq : ds₁ = ds₂) (h : Forall₂ (· < ·) is ds₁) :
                                                            /-
                                                              ds₁ ds₂ is : List Nat
                                                              eq : Eq ds₁ ds₂
                                                              h : List.Forall₂ (fun x1 x2 => LT.lt x1 x2) is ds₁
                                                              ⊢ Eq (↑(cast ⋯ ⟨is, h⟩)) is
                                                            -/
    (cast (congr_arg HolorIndex eq) ⟨is, h⟩).val = is := by subst eq; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


def assocRight : HolorIndex (ds₁ ++ ds₂ ++ ds₃) → HolorIndex (ds₁ ++ (ds₂ ++ ds₃)) :=
  cast (congr_arg HolorIndex (append_assoc ds₁ ds₂ ds₃))


def assocLeft : HolorIndex (ds₁ ++ (ds₂ ++ ds₃)) → HolorIndex (ds₁ ++ ds₂ ++ ds₃) :=
  cast (congr_arg HolorIndex (append_assoc ds₁ ds₂ ds₃).symm)


theorem take_take : ∀ t : HolorIndex (ds₁ ++ ds₂ ++ ds₃), t.assocRight.take = t.take.take
  | ⟨is, h⟩ =>
    Subtype.eq <| by
      /-
        ds₁ ds₂ ds₃ is : List Nat
        h : List.Forall₂ (fun x1 x2 => LT.lt x1 x2) is (HAppend.hAppend (HAppend.hAppe …
        ⊢ Eq ↑(HolorIndex.assocRight ⟨is, h⟩).take ↑(HolorIndex.take ⟨is, h⟩).take
      -/
      simp [assocRight, take, cast_type, List.take_take, Nat.le_add_right, min_eq_left]
      /-
        🎉 no goals
      -/


theorem drop_take : ∀ t : HolorIndex (ds₁ ++ ds₂ ++ ds₃), t.assocRight.drop.take = t.take.drop
                              /-
                                ds₁ ds₂ ds₃ is : List Nat
                                h : List.Forall₂ (fun x1 x2 => LT.lt x1 x2) is (HAppend.hAppend (HAppend.hAppe …
                                ⊢ Eq ↑(HolorIndex.assocRight ⟨is, h⟩).drop.take ↑(HolorIndex.take ⟨is, h⟩).drop
                              -/
  | ⟨is, h⟩ => Subtype.eq (by simp [assocRight, take, drop, cast_type, List.drop_take])
                              /-
                                🎉 no goals
                              -/


theorem drop_drop : ∀ t : HolorIndex (ds₁ ++ ds₂ ++ ds₃), t.assocRight.drop.drop = t.drop
                              /-
                                ds₁ ds₂ ds₃ is : List Nat
                                h : List.Forall₂ (fun x1 x2 => LT.lt x1 x2) is (HAppend.hAppend (HAppend.hAppe …
                                ⊢ Eq ↑(HolorIndex.assocRight ⟨is, h⟩).drop.drop ↑(HolorIndex.drop ⟨is, h⟩)
                              -/
  | ⟨is, h⟩ => Subtype.eq (by simp [add_comm, assocRight, drop, cast_type, List.drop_drop])
                              /-
                                🎉 no goals
                              -/


/-- Holor (indexed collections of tensor coefficients) -/
def Holor (α : Type u) (ds : List ℕ) :=
  HolorIndex ds → α


instance [Inhabited α] : Inhabited (Holor α ds) :=
  ⟨fun _ => default⟩


instance [Zero α] : Zero (Holor α ds) :=
  ⟨fun _ => 0⟩


instance [Add α] : Add (Holor α ds) :=
  ⟨fun x y t => x t + y t⟩


instance [Neg α] : Neg (Holor α ds) :=
  ⟨fun a t => -a t⟩


instance [AddSemigroup α] : AddSemigroup (Holor α ds) := Pi.addSemigroup


instance [AddCommSemigroup α] : AddCommSemigroup (Holor α ds) := Pi.addCommSemigroup


instance [AddMonoid α] : AddMonoid (Holor α ds) := Pi.addMonoid


instance [AddCommMonoid α] : AddCommMonoid (Holor α ds) := Pi.addCommMonoid


instance [AddGroup α] : AddGroup (Holor α ds) := Pi.addGroup


instance [AddCommGroup α] : AddCommGroup (Holor α ds) := Pi.addCommGroup

-- scalar product

instance [Mul α] : SMul α (Holor α ds) :=
  ⟨fun a x => fun t => a * x t⟩


instance [Semiring α] : Module α (Holor α ds) := Pi.module _ _ _


/-- The tensor product of two holors. -/
def mul [Mul α] (x : Holor α ds₁) (y : Holor α ds₂) : Holor α (ds₁ ++ ds₂) := fun t =>
  x t.take * y t.drop


local infixl:70 " ⊗ " => mul


theorem cast_type (eq : ds₁ = ds₂) (a : Holor α ds₁) :
    cast (congr_arg (Holor α) eq) a = fun t => a (cast (congr_arg HolorIndex eq.symm) t) := by
  /-
    α : Type
    ds₁ ds₂ : List Nat
    eq : Eq ds₁ ds₂
    a : Holor α ds₁
    ⊢ Eq (cast ⋯ a) fun t => a (cast ⋯ t)
  -/
  subst eq; rfl
            /-
              🎉 no goals
            -/


def assocRight : Holor α (ds₁ ++ ds₂ ++ ds₃) → Holor α (ds₁ ++ (ds₂ ++ ds₃)) :=
  cast (congr_arg (Holor α) (append_assoc ds₁ ds₂ ds₃))


def assocLeft : Holor α (ds₁ ++ (ds₂ ++ ds₃)) → Holor α (ds₁ ++ ds₂ ++ ds₃) :=
  cast (congr_arg (Holor α) (append_assoc ds₁ ds₂ ds₃).symm)


theorem mul_assoc0 [Semigroup α] (x : Holor α ds₁) (y : Holor α ds₂) (z : Holor α ds₃) :
    x ⊗ y ⊗ z = (x ⊗ (y ⊗ z)).assocLeft :=
  funext fun t : HolorIndex (ds₁ ++ ds₂ ++ ds₃) => by
    /-
      α : Type
      ds₁ ds₂ ds₃ : List Nat
      inst✝ : Semigroup α
      x : Holor α ds₁
      y : Holor α ds₂
      z : Holor α ds₃
      t : HolorIndex (HAppend.hAppend (HAppend.hAppend ds₁ ds₂) ds₃)
      ⊢ Eq ((x.mul y).mul z t) ((x.mul (y.mul z)).assocLeft t)
    -/
    rw [assocLeft]
    /-
      α : Type
      ds₁ ds₂ ds₃ : List Nat
      inst✝ : Semigroup α
      x : Holor α ds₁
      y : Holor α ds₂
      z : Holor α ds₃
      t : HolorIndex (HAppend.hAppend (HAppend.hAppend ds₁ ds₂) ds₃)
      ⊢ Eq ((x.mul y).mul z t) (cast ⋯ (x.mul (y.mul z)) t)
    -/
    unfold mul
    rw [mul_assoc, ← HolorIndex.take_take, ← HolorIndex.drop_take, ← HolorIndex.drop_drop,
      cast_type]
      /-
        α : Type
        ds₁ ds₂ ds₃ : List Nat
        inst✝ : Semigroup α
        x : Holor α ds₁
        y : Holor α ds₂
        z : Holor α ds₃
        t : HolorIndex (HAppend.hAppend (HAppend.hAppend ds₁ ds₂) ds₃)
        ⊢ Eq (HMul.hMul (x t.assocRight.take) (HMul.hMul (y t.assocRight.drop.take) (z …
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case eq
      α : Type
      ds₁ ds₂ ds₃ : List Nat
      inst✝ : Semigroup α
      x : Holor α ds₁
      y : Holor α ds₂
      z : Holor α ds₃
      t : HolorIndex (HAppend.hAppend (HAppend.hAppend ds₁ ds₂) ds₃)
      ⊢ Eq (HAppend.hAppend ds₁ (HAppend.hAppend ds₂ ds₃)) (HAppend.hAppend (HAppend …
    -/
    rw [append_assoc]
    /-
      🎉 no goals
    -/


theorem mul_assoc [Semigroup α] (x : Holor α ds₁) (y : Holor α ds₂) (z : Holor α ds₃) :
                                                  /-
                                                    α : Type
                                                    ds₁ ds₂ ds₃ : List Nat
                                                    inst✝ : Semigroup α
                                                    x : Holor α ds₁
                                                    y : Holor α ds₂
                                                    z : Holor α ds₃
                                                    ⊢ HEq ((x.mul y).mul z) (x.mul (y.mul z))
                                                  -/
    HEq (mul (mul x y) z) (mul x (mul y z)) := by simp [cast_heq, mul_assoc0, assocLeft]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem mul_left_distrib [Distrib α] (x : Holor α ds₁) (y : Holor α ds₂) (z : Holor α ds₂) :
    x ⊗ (y + z) = x ⊗ y + x ⊗ z := funext fun t => left_distrib (x t.take) (y t.drop) (z t.drop)


theorem mul_right_distrib [Distrib α] (x : Holor α ds₁) (y : Holor α ds₁) (z : Holor α ds₂) :
    (x + y) ⊗ z = x ⊗ z + y ⊗ z := funext fun t => add_mul (x t.take) (y t.take) (z t.drop)


@[simp]
nonrec theorem zero_mul {α : Type} [Ring α] (x : Holor α ds₂) : (0 : Holor α ds₁) ⊗ x = 0 :=
  funext fun t => zero_mul (x (HolorIndex.drop t))


@[simp]
nonrec theorem mul_zero {α : Type} [Ring α] (x : Holor α ds₁) : x ⊗ (0 : Holor α ds₂) = 0 :=
  funext fun t => mul_zero (x (HolorIndex.take t))


theorem mul_scalar_mul [Monoid α] (x : Holor α []) (y : Holor α ds) :
    x ⊗ y = x ⟨[], Forall₂.nil⟩ • y := by
  simp (config := { unfoldPartialApp := true }) [mul, SMul.smul, HolorIndex.take, HolorIndex.drop,
    HSMul.hSMul]

-- holor slices

/-- A slice is a subholor consisting of all entries with initial index i. -/
def slice (x : Holor α (d :: ds)) (i : ℕ) (h : i < d) : Holor α ds := fun is : HolorIndex ds =>
  x ⟨i :: is.1, Forall₂.cons h is.2⟩


/-- The 1-dimensional "unit" holor with 1 in the `j`th position. -/
def unitVec [Monoid α] [AddMonoid α] (d : ℕ) (j : ℕ) : Holor α [d] := fun ti =>
  if ti.1 = [j] then 1 else 0


theorem holor_index_cons_decomp (p : HolorIndex (d :: ds) → Prop) :
    ∀ t : HolorIndex (d :: ds),
                                                   /-
                                                     α : Type
                                                     d : Nat
                                                     ds ds₁ ds₂ ds₃ : List Nat
                                                     p : HolorIndex (List.cons d ds) → Prop
                                                     t : HolorIndex (List.cons d ds)
                                                     i : Nat
                                                     is : List Nat
                                                     h : Eq (↑t) (List.cons i is)
                                                     ⊢ List.Forall₂ (fun x1 x2 => LT.lt x1 x2) (List.cons i is) (List.cons d ds)
                                                   -/
      (∀ i is, ∀ h : t.1 = i :: is, p ⟨i :: is, by rw [← h]; exact t.2⟩) → p t
                                                             /-
                                                               🎉 no goals
                                                             -/
  | ⟨[], hforall₂⟩, _ => absurd (forall₂_nil_left_iff.1 hforall₂) (cons_ne_nil d ds)
  | ⟨i :: is, _⟩, hp => hp i is rfl


/-- Two holors are equal if all their slices are equal. -/
theorem slice_eq (x : Holor α (d :: ds)) (y : Holor α (d :: ds)) (h : slice x = slice y) : x = y :=
  funext fun t : HolorIndex (d :: ds) =>
    holor_index_cons_decomp (fun t => x t = y t) t fun i is hiis =>
                                                               /-
                                                                 α : Type
                                                                 d : Nat
                                                                 ds : List Nat
                                                                 x y : Holor α (List.cons d ds)
                                                                 h : Eq x.slice y.slice
                                                                 t : HolorIndex (List.cons d ds)
                                                                 i : Nat
                                                                 is : List Nat
                                                                 hiis : Eq (↑t) (List.cons i is)
                                                                 ⊢ List.Forall₂ (fun x1 x2 => LT.lt x1 x2) (List.cons i is) (List.cons d ds)
                                                               -/
      have hiisdds : Forall₂ (· < ·) (i :: is) (d :: ds) := by rw [← hiis]; exact t.2
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
      have hid : i < d := (forall₂_cons.1 hiisdds).1
      have hisds : Forall₂ (· < ·) is ds := (forall₂_cons.1 hiisdds).2
      calc
        x ⟨i :: is, _⟩ = slice x i hid ⟨is, hisds⟩ := congr_arg x (Subtype.eq rfl)
                                            /-
                                              α : Type
                                              d : Nat
                                              ds : List Nat
                                              x y : Holor α (List.cons d ds)
                                              h : Eq x.slice y.slice
                                              t : HolorIndex (List.cons d ds)
                                              i : Nat
                                              is : List Nat
                                              hiis : Eq (↑t) (List.cons i is)
                                              hiisdds : List.Forall₂ (fun x1 x2 => LT.lt x1 x2) (List.cons i is) (List.cons  …
                                              hid : LT.lt i d
                                              hisds : List.Forall₂ (fun x1 x2 => LT.lt x1 x2) is ds
                                              ⊢ Eq (x.slice i hid ⟨is, hisds⟩) (y.slice i hid ⟨is, hisds⟩)
                                            -/
        _ = slice y i hid ⟨is, hisds⟩ := by rw [h]
                                            /-
                                              🎉 no goals
                                            -/
        _ = y ⟨i :: is, _⟩ := congr_arg y (Subtype.eq rfl)


theorem slice_unitVec_mul [Ring α] {i : ℕ} {j : ℕ} (hid : i < d) (x : Holor α ds) :
    slice (unitVec d j ⊗ x) i hid = if i = j then x else 0 :=
  funext fun t : HolorIndex ds =>
                         /-
                           α : Type
                           d : Nat
                           ds : List Nat
                           inst✝ : Ring α
                           i j : Nat
                           hid : LT.lt i d
                           x : Holor α ds
                           t : HolorIndex ds
                           h : Eq i j
                           ⊢ Eq (((Holor.unitVec d j).mul x).slice i hid t) (ite (Eq i j) x 0 t)
                         -/
    if h : i = j then by simp [slice, mul, HolorIndex.take, unitVec, HolorIndex.drop, h]
                         /-
                           🎉 no goals
                         -/
            /-
              α : Type
              d : Nat
              ds : List Nat
              inst✝ : Ring α
              i j : Nat
              hid : LT.lt i d
              x : Holor α ds
              t : HolorIndex ds
              h : Not (Eq i j)
              ⊢ Eq (((Holor.unitVec d j).mul x).slice i hid t) (ite (Eq i j) x 0 t)
            -/
    else by simp [slice, mul, HolorIndex.take, unitVec, HolorIndex.drop, h]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem slice_add [Add α] (i : ℕ) (hid : i < d) (x : Holor α (d :: ds)) (y : Holor α (d :: ds)) :
    slice x i hid + slice y i hid = slice (x + y) i hid :=
                     /-
                       α : Type
                       d : Nat
                       ds : List Nat
                       inst✝ : Add α
                       i : Nat
                       hid : LT.lt i d
                       x y : Holor α (List.cons d ds)
                       t : HolorIndex ds
                       ⊢ Eq (HAdd.hAdd (x.slice i hid) (y.slice i hid) t) ((HAdd.hAdd x y).slice i hi …
                     -/
  funext fun t => by simp [slice, (· + ·), Add.add]
                     /-
                       🎉 no goals
                     -/


theorem slice_zero [Zero α] (i : ℕ) (hid : i < d) : slice (0 : Holor α (d :: ds)) i hid = 0 :=
  rfl


theorem slice_sum [AddCommMonoid α] {β : Type} (i : ℕ) (hid : i < d) (s : Finset β)
    (f : β → Holor α (d :: ds)) : (∑ x ∈ s, slice (f x) i hid) = slice (∑ x ∈ s, f x) i hid := by
  /-
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : AddCommMonoid α
    β : Type
    i : Nat
    hid : LT.lt i d
    s : Finset β
    f : β → Holor α (List.cons d ds)
    ⊢ Eq (s.sum fun x => (f x).slice i hid) ((s.sum fun x => f x).slice i hid)
  -/
  letI := Classical.decEq β
  /-
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : AddCommMonoid α
    β : Type
    i : Nat
    hid : LT.lt i d
    s : Finset β
    f : β → Holor α (List.cons d ds)
    this : DecidableEq β := Classical.decEq β
    ⊢ Eq (s.sum fun x => (f x).slice i hid) ((s.sum fun x => f x).slice i hid)
  -/
  refine Finset.induction_on s ?_ ?_
    /-
      case refine_1
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : AddCommMonoid α
      β : Type
      i : Nat
      hid : LT.lt i d
      s : Finset β
      f : β → Holor α (List.cons d ds)
      this : DecidableEq β := Classical.decEq β
      ⊢ Eq (EmptyCollection.emptyCollection.sum fun x => (f x).slice i hid) ((EmptyC …
    -/
  · simp [slice_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : AddCommMonoid α
      β : Type
      i : Nat
      hid : LT.lt i d
      s : Finset β
      f : β → Holor α (List.cons d ds)
      this : DecidableEq β := Classical.decEq β
      ⊢ ∀ ⦃a : β⦄ {s : Finset β}, Not (Membership.mem s a) → Eq (s.sum fun x => (f x …
    -/
  · intro _ _ h_not_in ih
    /-
      case refine_2
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : AddCommMonoid α
      β : Type
      i : Nat
      hid : LT.lt i d
      s : Finset β
      f : β → Holor α (List.cons d ds)
      this : DecidableEq β := Classical.decEq β
      a✝ : β
      s✝ : Finset β
      h_not_in : Not (Membership.mem s✝ a✝)
      ih : Eq (s✝.sum fun x => (f x).slice i hid) ((s✝.sum fun x => f x).slice i hid)
      ⊢ Eq ((Insert.insert a✝ s✝).sum fun x => (f x).slice i hid) (((Insert.insert a …
    -/
    rw [Finset.sum_insert h_not_in, ih, slice_add, Finset.sum_insert h_not_in]
    /-
      🎉 no goals
    -/


/-- The original holor can be recovered from its slices by multiplying with unit vectors and
summing up. -/
@[simp]
theorem sum_unitVec_mul_slice [Ring α] (x : Holor α (d :: ds)) :
    (∑ i ∈ (Finset.range d).attach,
        unitVec d i ⊗ slice x i (Nat.succ_le_of_lt (Finset.mem_range.1 i.prop))) =
      x := by
  /-
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : Ring α
    x : Holor α (List.cons d ds)
    ⊢ Eq ((Finset.range d).attach.sum fun i => (Holor.unitVec d ↑i).mul (x.slice ↑ …
  -/
  apply slice_eq _ _ _
  /-
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : Ring α
    x : Holor α (List.cons d ds)
    ⊢ Eq ((Finset.range d).attach.sum fun i => (Holor.unitVec d ↑i).mul (x.slice ↑ …
  -/
  ext i hid
  /-
    case h.h
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : Ring α
    x : Holor α (List.cons d ds)
    i : Nat
    hid : LT.lt i d
    ⊢ Eq (((Finset.range d).attach.sum fun i => (Holor.unitVec d ↑i).mul (x.slice  …
  -/
  rw [← slice_sum]
  /-
    case h.h
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : Ring α
    x : Holor α (List.cons d ds)
    i : Nat
    hid : LT.lt i d
    ⊢ Eq ((Finset.range d).attach.sum fun x_1 => ((Holor.unitVec d ↑x_1).mul (x.sl …
  -/
  simp only [slice_unitVec_mul hid]
  /-
    case h.h
    α : Type
    d : Nat
    ds : List Nat
    inst✝ : Ring α
    x : Holor α (List.cons d ds)
    i : Nat
    hid : LT.lt i d
    ⊢ Eq ((Finset.range d).attach.sum fun x_1 => ite (Eq i ↑x_1) (x.slice ↑x_1 ⋯)  …
  -/
  rw [Finset.sum_eq_single (Subtype.mk i <| Finset.mem_range.2 hid)]
    /-
      case h.h
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      ⊢ Eq (ite (Eq i ↑⟨i, ⋯⟩) (x.slice ↑⟨i, ⋯⟩ ⋯) 0) (x.slice i hid)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.h.h₀
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      ⊢ ∀ (b : Subtype (Membership.mem (Finset.range d))), Membership.mem (Finset.ra …
    -/
  · intro (b : { x // x ∈ Finset.range d }) (_ : b ∈ (Finset.range d).attach) (hbi : b ≠ ⟨i, _⟩)
    /-
      case h.h.h₀
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      b : Subtype fun x => Membership.mem (Finset.range d) x
      x✝ : Membership.mem (Finset.range d).attach b
      hbi : Ne b ⟨i, ⋯⟩
      ⊢ Eq (ite (Eq i ↑b) (x.slice ↑b ⋯) 0) 0
    -/
    have hbi' : i ≠ b := by simpa only [Ne, Subtype.ext_iff, Subtype.coe_mk] using hbi.symm
    /-
      case h.h.h₀
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      b : Subtype fun x => Membership.mem (Finset.range d) x
      x✝ : Membership.mem (Finset.range d).attach b
      hbi : Ne b ⟨i, ⋯⟩
      hbi' : Ne i ↑b
      ⊢ Eq (ite (Eq i ↑b) (x.slice ↑b ⋯) 0) 0
    -/
    simp [hbi']
    /-
      🎉 no goals
    -/
    /-
      case h.h.h₁
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      ⊢ Not (Membership.mem (Finset.range d).attach ⟨i, ⋯⟩) → Eq (ite (Eq i ↑⟨i, ⋯⟩) …
    -/
  · intro (hid' : Subtype.mk i _ ∉ Finset.attach (Finset.range d))
    /-
      case h.h.h₁
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      hid' : Not (Membership.mem (Finset.range d).attach ⟨i, ⋯⟩)
      ⊢ Eq (ite (Eq i ↑⟨i, ⋯⟩) (x.slice ↑⟨i, ⋯⟩ ⋯) 0) 0
    -/
    exfalso
    /-
      case h.h.h₁
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      x : Holor α (List.cons d ds)
      i : Nat
      hid : LT.lt i d
      hid' : Not (Membership.mem (Finset.range d).attach ⟨i, ⋯⟩)
      ⊢ False
    -/
    exact absurd (Finset.mem_attach _ _) hid'
    /-
      🎉 no goals
    -/

-- CP rank

/-- `CPRankMax1 x` means `x` has CP rank at most 1, that is,
  it is the tensor product of 1-dimensional holors. -/
inductive CPRankMax1 [Mul α] : ∀ {ds}, Holor α ds → Prop
  | nil (x : Holor α []) : CPRankMax1 x
  | cons {d} {ds} (x : Holor α [d]) (y : Holor α ds) : CPRankMax1 y → CPRankMax1 (x ⊗ y)


/-- `CPRankMax N x` means `x` has CP rank at most `N`, that is,
  it can be written as the sum of N holors of rank at most 1. -/
inductive CPRankMax [Mul α] [AddMonoid α] : ℕ → ∀ {ds}, Holor α ds → Prop
  | zero {ds} : CPRankMax 0 (0 : Holor α ds)
  | succ (n) {ds} (x : Holor α ds) (y : Holor α ds) :
    CPRankMax1 x → CPRankMax n y → CPRankMax (n + 1) (x + y)


theorem cprankMax_nil [Monoid α] [AddMonoid α] (x : Holor α nil) : CPRankMax 1 x := by
  /-
    α : Type
    inst✝¹ : Monoid α
    inst✝ : AddMonoid α
    x : Holor α List.nil
    ⊢ Holor.CPRankMax 1 x
  -/
  have h := CPRankMax.succ 0 x 0 (CPRankMax1.nil x) CPRankMax.zero
  /-
    α : Type
    inst✝¹ : Monoid α
    inst✝ : AddMonoid α
    x : Holor α List.nil
    h : Holor.CPRankMax (HAdd.hAdd 0 1) (HAdd.hAdd x 0)
    ⊢ Holor.CPRankMax 1 x
  -/
  rwa [add_zero x, zero_add] at h
  /-
    🎉 no goals
  -/


theorem cprankMax_1 [Monoid α] [AddMonoid α] {x : Holor α ds} (h : CPRankMax1 x) :
    CPRankMax 1 x := by
  /-
    α : Type
    ds : List Nat
    inst✝¹ : Monoid α
    inst✝ : AddMonoid α
    x : Holor α ds
    h : x.CPRankMax1
    ⊢ Holor.CPRankMax 1 x
  -/
  have h' := CPRankMax.succ 0 x 0 h CPRankMax.zero
  /-
    α : Type
    ds : List Nat
    inst✝¹ : Monoid α
    inst✝ : AddMonoid α
    x : Holor α ds
    h : x.CPRankMax1
    h' : Holor.CPRankMax (HAdd.hAdd 0 1) (HAdd.hAdd x 0)
    ⊢ Holor.CPRankMax 1 x
  -/
  rwa [zero_add, add_zero] at h'
  /-
    🎉 no goals
  -/


theorem cprankMax_add [Monoid α] [AddMonoid α] :
    ∀ {m : ℕ} {n : ℕ} {x : Holor α ds} {y : Holor α ds},
      CPRankMax m x → CPRankMax n y → CPRankMax (m + n) (x + y)
  | 0, n, x, y, hx, hy => by
    match hx with
    | CPRankMax.zero => simp only [zero_add, hy]
  | m + 1, n, _, y, CPRankMax.succ _ x₁ x₂ hx₁ hx₂, hy => by
    /-
      α : Type
      ds : List Nat
      inst✝¹ : Monoid α
      inst✝ : AddMonoid α
      m n : Nat
      y x₁ x₂ : Holor α ds
      hx₁ : x₁.CPRankMax1
      hx₂ : Holor.CPRankMax m x₂
      hy : Holor.CPRankMax n y
      ⊢ Holor.CPRankMax (HAdd.hAdd (HAdd.hAdd m 1) n) (HAdd.hAdd (HAdd.hAdd x₁ x₂) y)
    -/
    simp only [add_comm, add_assoc]
    /-
      α : Type
      ds : List Nat
      inst✝¹ : Monoid α
      inst✝ : AddMonoid α
      m n : Nat
      y x₁ x₂ : Holor α ds
      hx₁ : x₁.CPRankMax1
      hx₂ : Holor.CPRankMax m x₂
      hy : Holor.CPRankMax n y
      ⊢ Holor.CPRankMax (HAdd.hAdd n (HAdd.hAdd m 1)) (HAdd.hAdd x₁ (HAdd.hAdd x₂ y))
    -/
    apply CPRankMax.succ
      /-
        case a
        α : Type
        ds : List Nat
        inst✝¹ : Monoid α
        inst✝ : AddMonoid α
        m n : Nat
        y x₁ x₂ : Holor α ds
        hx₁ : x₁.CPRankMax1
        hx₂ : Holor.CPRankMax m x₂
        hy : Holor.CPRankMax n y
        ⊢ x₁.CPRankMax1
      -/
    · assumption
      /-
        🎉 no goals
      -/
    · -- Porting note: Single line is added.
      /-
        case a
        α : Type
        ds : List Nat
        inst✝¹ : Monoid α
        inst✝ : AddMonoid α
        m n : Nat
        y x₁ x₂ : Holor α ds
        hx₁ : x₁.CPRankMax1
        hx₂ : Holor.CPRankMax m x₂
        hy : Holor.CPRankMax n y
        ⊢ Holor.CPRankMax (n.add m) (HAdd.hAdd x₂ y)
      -/
      simp only [Nat.add_eq, add_zero, add_comm n m]
      /-
        case a
        α : Type
        ds : List Nat
        inst✝¹ : Monoid α
        inst✝ : AddMonoid α
        m n : Nat
        y x₁ x₂ : Holor α ds
        hx₁ : x₁.CPRankMax1
        hx₂ : Holor.CPRankMax m x₂
        hy : Holor.CPRankMax n y
        ⊢ Holor.CPRankMax (HAdd.hAdd m n) (HAdd.hAdd x₂ y)
      -/
      exact cprankMax_add hx₂ hy
      /-
        🎉 no goals
      -/


theorem cprankMax_mul [Ring α] :
    ∀ (n : ℕ) (x : Holor α [d]) (y : Holor α ds), CPRankMax n y → CPRankMax n (x ⊗ y)
                                  /-
                                    α : Type
                                    d : Nat
                                    ds : List Nat
                                    inst✝ : Ring α
                                    x : Holor α (List.cons d List.nil)
                                    ⊢ Holor.CPRankMax 0 (x.mul 0)
                                  -/
  | 0, x, _, CPRankMax.zero => by simp [mul_zero x, CPRankMax.zero]
                                  /-
                                    🎉 no goals
                                  -/
  | n + 1, x, _, CPRankMax.succ _ y₁ y₂ hy₁ hy₂ => by
    /-
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      n : Nat
      x : Holor α (List.cons d List.nil)
      y₁ y₂ : Holor α ds
      hy₁ : y₁.CPRankMax1
      hy₂ : Holor.CPRankMax n y₂
      ⊢ Holor.CPRankMax (HAdd.hAdd n 1) (x.mul (HAdd.hAdd y₁ y₂))
    -/
    rw [mul_left_distrib]
    /-
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      n : Nat
      x : Holor α (List.cons d List.nil)
      y₁ y₂ : Holor α ds
      hy₁ : y₁.CPRankMax1
      hy₂ : Holor.CPRankMax n y₂
      ⊢ Holor.CPRankMax (HAdd.hAdd n 1) (HAdd.hAdd (x.mul y₁) (x.mul y₂))
    -/
    rw [Nat.add_comm]
    /-
      α : Type
      d : Nat
      ds : List Nat
      inst✝ : Ring α
      n : Nat
      x : Holor α (List.cons d List.nil)
      y₁ y₂ : Holor α ds
      hy₁ : y₁.CPRankMax1
      hy₂ : Holor.CPRankMax n y₂
      ⊢ Holor.CPRankMax (HAdd.hAdd 1 n) (HAdd.hAdd (x.mul y₁) (x.mul y₂))
    -/
    apply cprankMax_add
      /-
        case a
        α : Type
        d : Nat
        ds : List Nat
        inst✝ : Ring α
        n : Nat
        x : Holor α (List.cons d List.nil)
        y₁ y₂ : Holor α ds
        hy₁ : y₁.CPRankMax1
        hy₂ : Holor.CPRankMax n y₂
        ⊢ Holor.CPRankMax 1 (x.mul y₁)
      -/
    · exact cprankMax_1 (CPRankMax1.cons _ _ hy₁)
      /-
        🎉 no goals
      -/
      /-
        case a
        α : Type
        d : Nat
        ds : List Nat
        inst✝ : Ring α
        n : Nat
        x : Holor α (List.cons d List.nil)
        y₁ y₂ : Holor α ds
        hy₁ : y₁.CPRankMax1
        hy₂ : Holor.CPRankMax n y₂
        ⊢ Holor.CPRankMax n (x.mul y₂)
      -/
    · exact cprankMax_mul _ x y₂ hy₂
      /-
        🎉 no goals
      -/


theorem cprankMax_sum [Ring α] {β} {n : ℕ} (s : Finset β) (f : β → Holor α ds) :
    (∀ x ∈ s, CPRankMax n (f x)) → CPRankMax (s.card * n) (∑ x ∈ s, f x) :=
  letI := Classical.decEq β
                            /-
                              α : Type
                              ds : List Nat
                              inst✝ : Ring α
                              β : Type u_1
                              n : Nat
                              s : Finset β
                              f : β → Holor α ds
                              this : DecidableEq β := Classical.decEq β
                              ⊢ (∀ (x : β), Membership.mem EmptyCollection.emptyCollection x → Holor.CPRankM …
                            -/
  Finset.induction_on s (by simp [CPRankMax.zero])
                            /-
                              🎉 no goals
                            -/
    (by
      /-
        α : Type
        ds : List Nat
        inst✝ : Ring α
        β : Type u_1
        n : Nat
        s : Finset β
        f : β → Holor α ds
        this : DecidableEq β := Classical.decEq β
        ⊢ ∀ ⦃a : β⦄ {s : Finset β}, Not (Membership.mem s a) → ((∀ (x : β), Membership …
      -/
      intro x s (h_x_notin_s : x ∉ s) ih h_cprank
      /-
        α : Type
        ds : List Nat
        inst✝ : Ring α
        β : Type u_1
        n : Nat
        s✝ : Finset β
        f : β → Holor α ds
        this : DecidableEq β := Classical.decEq β
        x : β
        s : Finset β
        h_x_notin_s : Not (Membership.mem s x)
        ih : (∀ (x : β), Membership.mem s x → Holor.CPRankMax n (f x)) → Holor.CPRankM …
        h_cprank : ∀ (x_1 : β), Membership.mem (Insert.insert x s) x_1 → Holor.CPRankM …
        ⊢ Holor.CPRankMax (HMul.hMul (Insert.insert x s).card n) ((Insert.insert x s). …
      -/
      simp only [Finset.sum_insert h_x_notin_s, Finset.card_insert_of_not_mem h_x_notin_s]
      /-
        α : Type
        ds : List Nat
        inst✝ : Ring α
        β : Type u_1
        n : Nat
        s✝ : Finset β
        f : β → Holor α ds
        this : DecidableEq β := Classical.decEq β
        x : β
        s : Finset β
        h_x_notin_s : Not (Membership.mem s x)
        ih : (∀ (x : β), Membership.mem s x → Holor.CPRankMax n (f x)) → Holor.CPRankM …
        h_cprank : ∀ (x_1 : β), Membership.mem (Insert.insert x s) x_1 → Holor.CPRankM …
        ⊢ Holor.CPRankMax (HMul.hMul (HAdd.hAdd s.card 1) n) (HAdd.hAdd (f x) (s.sum f …
      -/
      rw [Nat.right_distrib]
      /-
        α : Type
        ds : List Nat
        inst✝ : Ring α
        β : Type u_1
        n : Nat
        s✝ : Finset β
        f : β → Holor α ds
        this : DecidableEq β := Classical.decEq β
        x : β
        s : Finset β
        h_x_notin_s : Not (Membership.mem s x)
        ih : (∀ (x : β), Membership.mem s x → Holor.CPRankMax n (f x)) → Holor.CPRankM …
        h_cprank : ∀ (x_1 : β), Membership.mem (Insert.insert x s) x_1 → Holor.CPRankM …
        ⊢ Holor.CPRankMax (HAdd.hAdd (HMul.hMul s.card n) (HMul.hMul 1 n)) (HAdd.hAdd  …
      -/
      simp only [Nat.one_mul, Nat.add_comm]
      have ih' : CPRankMax (Finset.card s * n) (∑ x ∈ s, f x) := by
        apply ih
        intro (x : β) (h_x_in_s : x ∈ s)
        simp only [h_cprank, Finset.mem_insert_of_mem, h_x_in_s]
      /-
        α : Type
        ds : List Nat
        inst✝ : Ring α
        β : Type u_1
        n : Nat
        s✝ : Finset β
        f : β → Holor α ds
        this : DecidableEq β := Classical.decEq β
        x : β
        s : Finset β
        h_x_notin_s : Not (Membership.mem s x)
        ih : (∀ (x : β), Membership.mem s x → Holor.CPRankMax n (f x)) → Holor.CPRankM …
        h_cprank : ∀ (x_1 : β), Membership.mem (Insert.insert x s) x_1 → Holor.CPRankM …
        ih' : Holor.CPRankMax (HMul.hMul s.card n) (s.sum fun x => f x)
        ⊢ Holor.CPRankMax (HAdd.hAdd n (HMul.hMul s.card n)) (HAdd.hAdd (f x) (s.sum f …
      -/
      exact cprankMax_add (h_cprank x (Finset.mem_insert_self x s)) ih')
      /-
        🎉 no goals
      -/


theorem cprankMax_upper_bound [Ring α] : ∀ {ds}, ∀ x : Holor α ds, CPRankMax ds.prod x
  | [], x => cprankMax_nil x
  | d :: ds, x => by
    have h_summands :
      ∀ i : { x // x ∈ Finset.range d },
        CPRankMax ds.prod (unitVec d i.1 ⊗ slice x i.1 (mem_range.1 i.2)) :=
      fun i => cprankMax_mul _ _ _ (cprankMax_upper_bound (slice x i.1 (mem_range.1 i.2)))
    have h_dds_prod : (List.cons d ds).prod = Finset.card (Finset.range d) * prod ds := by
      simp [Finset.card_range]
    have :
      CPRankMax (Finset.card (Finset.attach (Finset.range d)) * prod ds)
        (∑ i ∈ Finset.attach (Finset.range d),
          unitVec d i.val ⊗ slice x i.val (mem_range.1 i.2)) :=
      cprankMax_sum (Finset.range d).attach _ fun i _ => h_summands i
    have h_cprankMax_sum :
      CPRankMax (Finset.card (Finset.range d) * prod ds)
        (∑ i ∈ Finset.attach (Finset.range d),
          unitVec d i.val ⊗ slice x i.val (mem_range.1 i.2)) := by rwa [Finset.card_attach] at this
    /-
      α : Type
      inst✝ : Ring α
      d : Nat
      ds : List Nat
      x : Holor α (List.cons d ds)
      h_summands : ∀ (i : Subtype fun x => Membership.mem (Finset.range d) x), Holor …
      h_dds_prod : Eq (List.cons d ds).prod (HMul.hMul (Finset.range d).card ds.prod)
      this : Holor.CPRankMax (HMul.hMul (Finset.range d).attach.card ds.prod) ((Fins …
      h_cprankMax_sum : Holor.CPRankMax (HMul.hMul (Finset.range d).card ds.prod) (( …
      ⊢ Holor.CPRankMax (List.cons d ds).prod x
    -/
    rw [← sum_unitVec_mul_slice x]
    /-
      α : Type
      inst✝ : Ring α
      d : Nat
      ds : List Nat
      x : Holor α (List.cons d ds)
      h_summands : ∀ (i : Subtype fun x => Membership.mem (Finset.range d) x), Holor …
      h_dds_prod : Eq (List.cons d ds).prod (HMul.hMul (Finset.range d).card ds.prod)
      this : Holor.CPRankMax (HMul.hMul (Finset.range d).attach.card ds.prod) ((Fins …
      h_cprankMax_sum : Holor.CPRankMax (HMul.hMul (Finset.range d).card ds.prod) (( …
      ⊢ Holor.CPRankMax (List.cons d ds).prod ((Finset.range d).attach.sum fun i =>  …
    -/
    rw [h_dds_prod]
    /-
      α : Type
      inst✝ : Ring α
      d : Nat
      ds : List Nat
      x : Holor α (List.cons d ds)
      h_summands : ∀ (i : Subtype fun x => Membership.mem (Finset.range d) x), Holor …
      h_dds_prod : Eq (List.cons d ds).prod (HMul.hMul (Finset.range d).card ds.prod)
      this : Holor.CPRankMax (HMul.hMul (Finset.range d).attach.card ds.prod) ((Fins …
      h_cprankMax_sum : Holor.CPRankMax (HMul.hMul (Finset.range d).card ds.prod) (( …
      ⊢ Holor.CPRankMax (HMul.hMul (Finset.range d).card ds.prod) ((Finset.range d). …
    -/
    exact h_cprankMax_sum
    /-
      🎉 no goals
    -/


/-- The CP rank of a holor `x`: the smallest N such that
  `x` can be written as the sum of N holors of rank at most 1. -/
noncomputable def cprank [Ring α] (x : Holor α ds) : Nat :=
  @Nat.find (fun n => CPRankMax n x) (Classical.decPred _) ⟨ds.prod, cprankMax_upper_bound x⟩


theorem cprank_upper_bound [Ring α] : ∀ {ds}, ∀ x : Holor α ds, cprank x ≤ ds.prod :=
  fun {ds} x =>
  letI := Classical.decPred fun n : ℕ => CPRankMax n x
  Nat.find_min' ⟨ds.prod, show (fun n => CPRankMax n x) ds.prod from cprankMax_upper_bound x⟩
    (cprankMax_upper_bound x)


