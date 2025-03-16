/-- `Vector α n` is the type of lists of length `n` with elements of type `α`. -/
def List.Vector (α : Type u) (n : ℕ) :=
  { l : List α // l.length = n }


instance [DecidableEq α] : DecidableEq (Vector α n) :=
  inferInstanceAs (DecidableEq {l : List α // l.length = n})


/-- The empty vector with elements of type `α` -/
@[match_pattern]
def nil : Vector α 0 :=
  ⟨[], rfl⟩


/-- If `a : α` and `l : Vector α n`, then `cons a l`, is the vector of length `n + 1`
whose first element is a and with l as the rest of the list. -/
@[match_pattern]
def cons : α → Vector α n → Vector α (Nat.succ n)
  | a, ⟨v, h⟩ => ⟨a :: v, congrArg Nat.succ h⟩



/-- The length of a vector. -/
@[reducible, nolint unusedArguments]
def length (_ : Vector α n) : ℕ :=
  n


/-- The first element of a vector with length at least `1`. -/
def head : Vector α (Nat.succ n) → α
  | ⟨a :: _, _⟩ => a


/-- The head of a vector obtained by prepending is the element prepended. -/
theorem head_cons (a : α) : ∀ v : Vector α n, head (cons a v) = a
  | ⟨_, _⟩ => rfl


/-- The tail of a vector, with an empty vector having empty tail. -/
def tail : Vector α n → Vector α (n - 1)
  | ⟨[], h⟩ => ⟨[], congrArg pred h⟩
  | ⟨_ :: v, h⟩ => ⟨v, congrArg pred h⟩


/-- The tail of a vector obtained by prepending is the vector prepended. to -/
theorem tail_cons (a : α) : ∀ v : Vector α n, tail (cons a v) = v
  | ⟨_, _⟩ => rfl


/-- Prepending the head of a vector to its tail gives the vector. -/
@[simp]
theorem cons_head_tail : ∀ v : Vector α (succ n), cons (head v) (tail v) = v
                  /-
                    α : Type u_1
                    n : Nat
                    h : Eq List.nil.length n.succ
                    ⊢ Eq (List.Vector.cons (List.Vector.head ⟨List.nil, h⟩) (List.Vector.tail ⟨Lis …
                  -/
  | ⟨[], h⟩ => by contradiction
                  /-
                    🎉 no goals
                  -/
  | ⟨_ :: _, _⟩ => rfl


/-- The list obtained from a vector. -/
def toList (v : Vector α n) : List α :=
  v.1


/-- nth element of a vector, indexed by a `Fin` type. -/
def get (l : Vector α n) (i : Fin n) : α :=
  l.1.get <| i.cast l.2.symm


/-- Appending a vector to another. -/
def append {n m : Nat} : Vector α n → Vector α m → Vector α (n + m)
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          σ : Type u_3
                                          φ : Type u_4
                                          n✝ : Nat
                                          p : α → Prop
                                          n m : Nat
                                          l₁ : List α
                                          h₁ : Eq l₁.length n
                                          l₂ : List α
                                          h₂ : Eq l₂.length m
                                          ⊢ Eq (HAppend.hAppend l₁ l₂).length (HAdd.hAdd n m)
                                        -/
  | ⟨l₁, h₁⟩, ⟨l₂, h₂⟩ => ⟨l₁ ++ l₂, by simp [*]⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- Elimination rule for `Vector`. -/
@[elab_as_elim]
def elim {α} {C : ∀ {n}, Vector α n → Sort u}
    (H : ∀ l : List α, C ⟨l, rfl⟩) {n : ℕ} : ∀ v : Vector α n, C v
  | ⟨l, h⟩ =>
    match n, h with
    | _, rfl => H l


/-- Map a vector under a function. -/
def map (f : α → β) : Vector α n → Vector β n
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  σ : Type u_3
                                  φ : Type u_4
                                  n : Nat
                                  p : α → Prop
                                  f : α → β
                                  l : List α
                                  h : Eq l.length n
                                  ⊢ Eq (List.map f l).length n
                                -/
  | ⟨l, h⟩ => ⟨List.map f l, by simp [*]⟩
                                /-
                                  🎉 no goals
                                -/


/-- A `nil` vector maps to a `nil` vector. -/
@[simp]
theorem map_nil (f : α → β) : map f nil = nil :=
  rfl


/-- `map` is natural with respect to `cons`. -/
@[simp]
theorem map_cons (f : α → β) (a : α) : ∀ v : Vector α n, map f (cons a v) = cons (f a) (map f v)
  | ⟨_, _⟩ => rfl


/-- Map a vector under a partial function. -/
def pmap (f : (a : α) → p a → β) :
    (v : Vector α n) → (∀ x ∈ v.toList, p x) → Vector β n
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          σ : Type u_3
                                          φ : Type u_4
                                          n : Nat
                                          p : α → Prop
                                          f : (a : α) → p a → β
                                          l : List α
                                          h : Eq l.length n
                                          hp : ∀ (x : α), Membership.mem (List.Vector.toList ⟨l, h⟩) x → p x
                                          ⊢ Eq (List.pmap f l hp).length n
                                        -/
  | ⟨l, h⟩, hp => ⟨List.pmap f l hp, by simp [h]⟩
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem pmap_nil (f : (a : α) → p a → β) (hp : ∀ x ∈ nil.toList, p x) :
    nil.pmap f hp = nil := rfl


/-- Mapping two vectors under a curried function of two variables. -/
def map₂ (f : α → β → φ) : Vector α n → Vector β n → Vector φ n
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                σ : Type u_3
                                                φ : Type u_4
                                                n : Nat
                                                p : α → Prop
                                                f : α → β → φ
                                                x : List α
                                                property✝¹ : Eq x.length n
                                                y : List β
                                                property✝ : Eq y.length n
                                                ⊢ Eq (List.zipWith f x y).length n
                                              -/
  | ⟨x, _⟩, ⟨y, _⟩ => ⟨List.zipWith f x y, by simp [*]⟩
                                              /-
                                                🎉 no goals
                                              -/


/-- Vector obtained by repeating an element. -/
def replicate (n : ℕ) (a : α) : Vector α n :=
  ⟨List.replicate n a, List.length_replicate n a⟩


/-- Drop `i` elements from a vector of length `n`; we can have `i > n`. -/
def drop (i : ℕ) : Vector α n → Vector α (n - i)
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   σ : Type u_3
                                   φ : Type u_4
                                   n : Nat
                                   p✝ : α → Prop
                                   i : Nat
                                   l : List α
                                   p : Eq l.length n
                                   ⊢ Eq (List.drop i l).length (HSub.hSub n i)
                                 -/
  | ⟨l, p⟩ => ⟨List.drop i l, by simp [*]⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- Take `i` elements from a vector of length `n`; we can have `i > n`. -/
def take (i : ℕ) : Vector α n → Vector α (min i n)
                                 /-
                                   α : Type u_1
                                   β : Type u_2
                                   σ : Type u_3
                                   φ : Type u_4
                                   n : Nat
                                   p✝ : α → Prop
                                   i : Nat
                                   l : List α
                                   p : Eq l.length n
                                   ⊢ Eq (List.take i l).length (Min.min i n)
                                 -/
  | ⟨l, p⟩ => ⟨List.take i l, by simp [*]⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- Remove the element at position `i` from a vector of length `n`. -/
def eraseIdx (i : Fin n) : Vector α n → Vector α (n - 1)
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         σ : Type u_3
                                         φ : Type u_4
                                         n : Nat
                                         p✝ : α → Prop
                                         i : Fin n
                                         l : List α
                                         p : Eq l.length n
                                         ⊢ Eq (l.eraseIdx ↑i).length (HSub.hSub n 1)
                                       -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  | ⟨l, p⟩ => ⟨List.eraseIdx l i.1, by rw [l.length_eraseIdx_of_lt] <;> rw [p]; exact i.2⟩
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[deprecated (since := "2024-05-04")] alias removeNth := eraseIdx


/-- Vector of length `n` from a function on `Fin n`. -/
def ofFn : ∀ {n}, (Fin n → α) → Vector α n
  | 0, _ => nil
  | _ + 1, f => cons (f 0) (ofFn fun i ↦ f i.succ)


/-- Create a vector from another with a provably equal length. -/
protected def congr {n m : ℕ} (h : n = m) : Vector α n → Vector α m
  | ⟨x, p⟩ => ⟨x, h ▸ p⟩


/-- Runs a function over a vector returning the intermediate results and a
final result.
-/
def mapAccumr (f : α → σ → σ × β) : Vector α n → σ → σ × Vector β n
  | ⟨x, px⟩, c =>
    let res := List.mapAccumr f x c
                      /-
                        α : Type u_1
                        β : Type u_2
                        σ : Type u_3
                        φ : Type u_4
                        n : Nat
                        p : α → Prop
                        f : α → σ → Prod σ β
                        x : List α
                        px : Eq x.length n
                        c : σ
                        res : Prod σ (List β) := List.mapAccumr f x c
                        ⊢ Eq res.snd.length n
                      -/
    ⟨res.1, res.2, by simp [*, res]⟩
                      /-
                        🎉 no goals
                      -/


/-- Runs a function over a pair of vectors returning the intermediate results and a
final result.
-/
def mapAccumr₂ (f : α → β → σ → σ × φ) : Vector α n → Vector β n → σ → σ × Vector φ n
  | ⟨x, px⟩, ⟨y, py⟩, c =>
    let res := List.mapAccumr₂ f x y c
                      /-
                        α : Type u_1
                        β : Type u_2
                        σ : Type u_3
                        φ : Type u_4
                        n : Nat
                        p : α → Prop
                        f : α → β → σ → Prod σ φ
                        x : List α
                        px : Eq x.length n
                        y : List β
                        py : Eq y.length n
                        c : σ
                        res : Prod σ (List φ) := List.mapAccumr₂ f x y c
                        ⊢ Eq res.snd.length n
                      -/
    ⟨res.1, res.2, by simp [*, res]⟩
                      /-
                        🎉 no goals
                      -/


/-- `shiftLeftFill v i` is the vector obtained by left-shifting `v` `i` times and padding with the
    `fill` argument. If `v.length < i` then this will return `replicate n fill`. -/
def shiftLeftFill (v : Vector α n) (i : ℕ) (fill : α) : Vector α n :=
                   /-
                     α : Type u_1
                     β : Type u_2
                     σ : Type u_3
                     φ : Type u_4
                     n : Nat
                     p : α → Prop
                     v : List.Vector α n
                     i : Nat
                     fill : α
                     ⊢ Eq (HAdd.hAdd (HSub.hSub n i) (Min.min n i)) n
                   -/
  Vector.congr (by simp) <|
                   /-
                     🎉 no goals
                   -/
    append (drop i v) (replicate (min n i) fill)


/-- `shiftRightFill v i` is the vector obtained by right-shifting `v` `i` times and padding with the
    `fill` argument. If `v.length < i` then this will return `replicate n fill`. -/
def shiftRightFill (v : Vector α n) (i : ℕ) (fill : α) : Vector α n :=
                   /-
                     α : Type u_1
                     β : Type u_2
                     σ : Type u_3
                     φ : Type u_4
                     n : Nat
                     p : α → Prop
                     v : List.Vector α n
                     i : Nat
                     fill : α
                     ⊢ Eq (HAdd.hAdd (Min.min n i) (Min.min (HSub.hSub n i) n)) n
                   -/
  Vector.congr (by omega) <| append (replicate (min n i) fill) (take (n - i) v)
                   /-
                     🎉 no goals
                   -/


/-- Vector is determined by the underlying list. -/
protected theorem eq {n : ℕ} : ∀ a1 a2 : Vector α n, toList a1 = toList a2 → a1 = a2
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


/-- A vector of length `0` is a `nil` vector. -/
protected theorem eq_nil (v : Vector α 0) : v = nil :=
  v.eq nil (List.eq_nil_of_length_eq_zero v.2)


/-- Vector of length from a list `v`
with witness that `v` has length `n` maps to `v` under `toList`. -/
@[simp]
theorem toList_mk (v : List α) (P : List.length v = n) : toList (Subtype.mk v P) = v :=
  rfl


/-- A nil vector maps to a nil list. -/
@[simp]
theorem toList_nil : toList nil = @List.nil α :=
  rfl


/-- The length of the list to which a vector of length `n` maps is `n`. -/
@[simp]
theorem toList_length (v : Vector α n) : (toList v).length = n :=
  v.2


/-- `toList` of `cons` of a vector and an element is
the `cons` of the list obtained by `toList` and the element -/
@[simp]
theorem toList_cons (a : α) (v : Vector α n) : toList (cons a v) = a :: toList v := by
  /-
    α : Type u_1
    n : Nat
    a : α
    v : List.Vector α n
    ⊢ Eq (List.Vector.cons a v).toList (List.cons a v.toList)
  -/
  cases v; rfl
           /-
             🎉 no goals
           -/


/-- Appending of vectors corresponds under `toList` to appending of lists. -/
@[simp]
theorem toList_append {n m : ℕ} (v : Vector α n) (w : Vector α m) :
    toList (append v w) = toList v ++ toList w := by
  /-
    α : Type u_1
    n m : Nat
    v : List.Vector α n
    w : List.Vector α m
    ⊢ Eq (v.append w).toList (HAppend.hAppend v.toList w.toList)
  -/
  cases v
  /-
    case mk
    α : Type u_1
    n m : Nat
    w : List.Vector α m
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq (List.Vector.append ⟨val✝, property✝⟩ w).toList (HAppend.hAppend (List.Ve …
  -/
  cases w
  /-
    case mk.mk
    α : Type u_1
    n m : Nat
    val✝¹ : List α
    property✝¹ : Eq val✝¹.length n
    val✝ : List α
    property✝ : Eq val✝.length m
    ⊢ Eq (List.Vector.append ⟨val✝¹, property✝¹⟩ ⟨val✝, property✝⟩).toList (HAppen …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `drop` of vectors corresponds under `toList` to `drop` of lists. -/
@[simp]
theorem toList_drop {n m : ℕ} (v : Vector α m) : toList (drop n v) = List.drop n (toList v) := by
  /-
    α : Type u_1
    n m : Nat
    v : List.Vector α m
    ⊢ Eq (List.Vector.drop n v).toList (List.drop n v.toList)
  -/
  cases v
  /-
    case mk
    α : Type u_1
    n m : Nat
    val✝ : List α
    property✝ : Eq val✝.length m
    ⊢ Eq (List.Vector.drop n ⟨val✝, property✝⟩).toList (List.drop n (List.Vector.t …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `take` of vectors corresponds under `toList` to `take` of lists. -/
@[simp]
theorem toList_take {n m : ℕ} (v : Vector α m) : toList (take n v) = List.take n (toList v) := by
  /-
    α : Type u_1
    n m : Nat
    v : List.Vector α m
    ⊢ Eq (List.Vector.take n v).toList (List.take n v.toList)
  -/
  cases v
  /-
    case mk
    α : Type u_1
    n m : Nat
    val✝ : List α
    property✝ : Eq val✝.length m
    ⊢ Eq (List.Vector.take n ⟨val✝, property✝⟩).toList (List.take n (List.Vector.t …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : GetElem (Vector α n) Nat α fun _ i => i < n where
  getElem := fun x i h => get x ⟨i, h⟩


lemma getElem_def (v : Vector α n) (i : ℕ) {hi : i < n} :
                           /-
                             α : Type u_1
                             β : Type u_2
                             σ : Type u_3
                             φ : Type u_4
                             n : Nat
                             p : α → Prop
                             v : List.Vector α n
                             i : Nat
                             hi : LT.lt i n
                             ⊢ LT.lt i v.toList.length
                           -/
    v[i] = v.toList[i]'(by simpa) := rfl
                           /-
                             🎉 no goals
                           -/


lemma toList_getElem (v : Vector α n) (i : ℕ) {hi : i < v.toList.length} :
                           /-
                             α : Type u_1
                             β : Type u_2
                             σ : Type u_3
                             φ : Type u_4
                             n : Nat
                             p : α → Prop
                             v : List.Vector α n
                             i : Nat
                             hi : LT.lt i v.toList.length
                             ⊢ LT.lt i n
                           -/
    v.toList[i] = v[i]'(by simp_all) := rfl
                           /-
                             🎉 no goals
                           -/


