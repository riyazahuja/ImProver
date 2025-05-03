/-- Alternate definition of `Vector` based on `Fin2`. -/
def Vector3 (α : Type u) (n : ℕ) : Type u :=
  Fin2 n → α


instance [Inhabited α] : Inhabited (Vector3 α n) where
  default := fun _ => default


/-- The empty vector -/
@[match_pattern]
def nil : Vector3 α 0 :=
  nofun


/-- The vector cons operation -/
@[match_pattern]
def cons (a : α) (v : Vector3 α n) : Vector3 α (n + 1) := fun i => by
  /-
    α : Type u_1
    m n : Nat
    a : α
    v : Vector3 α n
    i : Fin2 (HAdd.hAdd n 1)
    ⊢ α
  -/
  refine i.cases' ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m n : Nat
      a : α
      v : Vector3 α n
      i : Fin2 (HAdd.hAdd n 1)
      ⊢ α
    -/
  · exact a
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m n : Nat
      a : α
      v : Vector3 α n
      i : Fin2 (HAdd.hAdd n 1)
      ⊢ Fin2 n → α
    -/
  · exact v
    /-
      🎉 no goals
    -/


scoped macro_rules | `([$l,*]) => `(expand_foldr% (h t => cons h t) nil [$(.ofElems l),*])

-- this is copied from `src/Init/NotationExtra.lean`

@[app_unexpander Vector3.nil] def unexpandNil : Lean.PrettyPrinter.Unexpander
  | `($(_)) => `([])

-- this is copied from `src/Init/NotationExtra.lean`

@[app_unexpander Vector3.cons] def unexpandCons : Lean.PrettyPrinter.Unexpander
  | `($(_) $x [])      => `([$x])
  | `($(_) $x [$xs,*]) => `([$x, $xs,*])
  | _                  => throw ()


@[inherit_doc]
scoped notation a " :: " b => cons a b


@[simp]
theorem cons_fz (a : α) (v : Vector3 α n) : (a :: v) fz = a :=
  rfl


@[simp]
theorem cons_fs (a : α) (v : Vector3 α n) (i) : (a :: v) (fs i) = v i :=
  rfl


/-- Get the `i`th element of a vector -/
abbrev nth (i : Fin2 n) (v : Vector3 α n) : α :=
  v i


/-- Construct a vector from a function on `Fin2`. -/
abbrev ofFn (f : Fin2 n → α) : Vector3 α n :=
  f


/-- Get the head of a nonempty vector. -/
def head (v : Vector3 α (n + 1)) : α :=
  v fz


/-- Get the tail of a nonempty vector. -/
def tail (v : Vector3 α (n + 1)) : Vector3 α n := fun i => v (fs i)


theorem eq_nil (v : Vector3 α 0) : v = [] :=
  funext fun i => nomatch i


theorem cons_head_tail (v : Vector3 α (n + 1)) : (head v :: tail v) = v :=
  funext fun i => Fin2.cases' rfl (fun _ => rfl) i


/-- Eliminator for an empty vector. -/
@[elab_as_elim]  -- Porting note: add `elab_as_elim`
def nilElim {C : Vector3 α 0 → Sort u} (H : C []) (v : Vector3 α 0) : C v := by
  /-
    α : Type u_1
    m n : Nat
    C : Vector3 α 0 → Sort u
    H : C Vector3.nil
    v : Vector3 α 0
    ⊢ C v
  -/
  rw [eq_nil v]; apply H
                 /-
                   🎉 no goals
                 -/


/-- Recursion principle for a nonempty vector. -/
@[elab_as_elim]  -- Porting note: add `elab_as_elim`
def consElim {C : Vector3 α (n + 1) → Sort u} (H : ∀ (a : α) (t : Vector3 α n), C (a :: t))
                                        /-
                                          α : Type u_1
                                          m n : Nat
                                          C : Vector3 α (HAdd.hAdd n 1) → Sort u
                                          H : (a : α) → (t : Vector3 α n) → C (Vector3.cons a t)
                                          v : Vector3 α (HAdd.hAdd n 1)
                                          ⊢ C v
                                        -/
    (v : Vector3 α (n + 1)) : C v := by rw [← cons_head_tail v]; apply H
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem consElim_cons {C H a t} : @consElim α n C H (a :: t) = H a t :=
  rfl


/-- Recursion principle with the vector as first argument. -/
@[elab_as_elim]
protected def recOn {C : ∀ {n}, Vector3 α n → Sort u} {n} (v : Vector3 α n) (H0 : C [])
    (Hs : ∀ {n} (a) (w : Vector3 α n), C w → C (a :: w)) : C v :=
  match n with
  | 0 => v.nilElim H0
  | _ + 1 => v.consElim fun a t => Hs a t (Vector3.recOn t H0 Hs)


@[simp]
theorem recOn_nil {C H0 Hs} : @Vector3.recOn α (@C) 0 [] H0 @Hs = H0 :=
  rfl


@[simp]
theorem recOn_cons {C H0 Hs n a v} :
    @Vector3.recOn α (@C) (n + 1) (a :: v) H0 @Hs = Hs a v (@Vector3.recOn α (@C) n v H0 @Hs) :=
  rfl


/-- Append two vectors -/
def append (v : Vector3 α m) (w : Vector3 α n) : Vector3 α (n + m) :=
  v.recOn w (fun a _ IH => a :: IH)


/--
A local infix notation for `Vector3.append`
-/
local infixl:65 " +-+ " => Vector3.append


@[simp]
theorem append_nil (w : Vector3 α n) : [] +-+ w = w :=
  rfl


@[simp]
theorem append_cons (a : α) (v : Vector3 α m) (w : Vector3 α n) : (a :: v) +-+ w = a :: v +-+ w :=
  rfl


@[simp]
theorem append_left :
    ∀ {m} (i : Fin2 m) (v : Vector3 α m) {n} (w : Vector3 α n), (v +-+ w) (left n i) = v i
                                                   /-
                                                     α : Type u_1
                                                     m : Nat
                                                     v : Vector3 α (HAdd.hAdd m 1)
                                                     x✝¹ : Nat
                                                     x✝ : Vector3 α x✝¹
                                                     a : α
                                                     _t : Vector3 α m
                                                     ⊢ Eq ((Vector3.cons a _t).append x✝ (Fin2.left x✝¹ Fin2.fz)) (Vector3.cons a _ …
                                                   -/
  | _, @fz m, v, _, _ => v.consElim fun a _t => by simp [*, left]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                     /-
                                                       α : Type u_1
                                                       m : Nat
                                                       i : Fin2 m
                                                       v : Vector3 α (HAdd.hAdd m 1)
                                                       n : Nat
                                                       w : Vector3 α n
                                                       _a : α
                                                       t : Vector3 α m
                                                       ⊢ Eq ((Vector3.cons _a t).append w (Fin2.left n i.fs)) (Vector3.cons _a t i.fs)
                                                     -/
  | _, @fs m i, v, n, w => v.consElim fun _a t => by simp [append_left, left]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem append_add :
    ∀ {m} (v : Vector3 α m) {n} (w : Vector3 α n) (i : Fin2 n), (v +-+ w) (add i m) = w i
  | 0, _, _, _, _ => rfl
                                                   /-
                                                     α : Type u_1
                                                     m : Nat
                                                     v : Vector3 α (HAdd.hAdd m 1)
                                                     n : Nat
                                                     w : Vector3 α n
                                                     i : Fin2 n
                                                     _a : α
                                                     t : Vector3 α m
                                                     ⊢ Eq ((Vector3.cons _a t).append w (i.add (HAdd.hAdd m 1))) (w i)
                                                   -/
  | m + 1, v, n, w, i => v.consElim fun _a t => by simp [append_add, add]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Insert `a` into `v` at index `i`. -/
def insert (a : α) (v : Vector3 α n) (i : Fin2 (n + 1)) : Vector3 α (n + 1) := fun j =>
  (a :: v) (insertPerm i j)


@[simp]
theorem insert_fz (a : α) (v : Vector3 α n) : insert a v fz = a :: v := by
  /-
    α : Type u_1
    n : Nat
    a : α
    v : Vector3 α n
    ⊢ Eq (Vector3.insert a v Fin2.fz) (Vector3.cons a v)
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  refine funext fun j => j.cases' ?_ ?_ <;> intros <;> rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem insert_fs (a : α) (b : α) (v : Vector3 α n) (i : Fin2 (n + 1)) :
    insert a (b :: v) (fs i) = b :: insert a v i :=
  funext fun j => by
    /-
      α : Type u_1
      n : Nat
      a b : α
      v : Vector3 α n
      i : Fin2 (HAdd.hAdd n 1)
      j : Fin2 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (Vector3.insert a (Vector3.cons b v) i.fs j) (Vector3.cons b (Vector3.ins …
    -/
                                       /-
                                         🎉 no goals
                                       -/
    refine j.cases' ?_ fun j => ?_ <;> simp [insert, insertPerm]
    /-
      case refine_2
      α : Type u_1
      n : Nat
      a b : α
      v : Vector3 α n
      i : Fin2 (HAdd.hAdd n 1)
      j✝ : Fin2 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      j : Fin2 (HAdd.hAdd n 1)
      ⊢ Eq (Vector3.cons a (Vector3.cons b v) (Fin2.cases'.match_1 (fun x => Fin2 (H …
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    refine Fin2.cases' ?_ ?_ (insertPerm i j) <;> simp [insertPerm]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem append_insert (a : α) (t : Vector3 α m) (v : Vector3 α n) (i : Fin2 (n + 1))
    (e : (n + 1) + m = (n + m) + 1) :
    insert a (t +-+ v) (Eq.recOn e (i.add m)) = Eq.recOn e (t +-+ insert a v i) := by
  /-
    α : Type u_1
    m n : Nat
    a : α
    t : Vector3 α m
    v : Vector3 α n
    i : Fin2 (HAdd.hAdd n 1)
    e : Eq (HAdd.hAdd (HAdd.hAdd n 1) m) (HAdd.hAdd (HAdd.hAdd n m) 1)
    ⊢ Eq (Vector3.insert a (t.append v) (Eq.recOn e (i.add m))) (Eq.recOn e (t.app …
  -/
  refine Vector3.recOn t (fun e => ?_) (@fun k b t IH _ => ?_) e
    /-
      case refine_1
      α : Type u_1
      m n : Nat
      a : α
      t : Vector3 α m
      v : Vector3 α n
      i : Fin2 (HAdd.hAdd n 1)
      e✝ : Eq (HAdd.hAdd (HAdd.hAdd n 1) m) (HAdd.hAdd (HAdd.hAdd n m) 1)
      e : Eq (HAdd.hAdd (HAdd.hAdd n 1) 0) (HAdd.hAdd (HAdd.hAdd n 0) 1)
      ⊢ Eq (Vector3.insert a (Vector3.nil.append v) (Eq.recOn e (i.add 0))) (Eq.recO …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    m n : Nat
    a : α
    t✝ : Vector3 α m
    v : Vector3 α n
    i : Fin2 (HAdd.hAdd n 1)
    e : Eq (HAdd.hAdd (HAdd.hAdd n 1) m) (HAdd.hAdd (HAdd.hAdd n m) 1)
    k : Nat
    b : α
    t : Vector3 α k
    IH : ∀ (x : Eq (HAdd.hAdd (HAdd.hAdd n 1) k) (HAdd.hAdd (HAdd.hAdd n k) 1)), E …
    x✝ : Eq (HAdd.hAdd (HAdd.hAdd n 1) (HAdd.hAdd k 1)) (HAdd.hAdd (HAdd.hAdd n (H …
    ⊢ Eq (Vector3.insert a ((Vector3.cons b t).append v) (Eq.recOn x✝ (i.add (HAdd …
  -/
  have e' : (n + 1) + k = (n + k) + 1 := by omega
  change
    insert a (b :: t +-+ v)
      (Eq.recOn (congr_arg (· + 1) e' : _ + 1 = _) (fs (add i k))) =
      Eq.recOn (congr_arg (· + 1) e' : _ + 1 = _) (b :: t +-+ insert a v i)
  rw [← (Eq.recOn e' rfl :
      fs (Eq.recOn e' (i.add k) : Fin2 ((n + k) + 1)) =
        Eq.recOn (congr_arg (· + 1) e' : _ + 1 = _) (fs (i.add k)))]
  /-
    case refine_2
    α : Type u_1
    m n : Nat
    a : α
    t✝ : Vector3 α m
    v : Vector3 α n
    i : Fin2 (HAdd.hAdd n 1)
    e : Eq (HAdd.hAdd (HAdd.hAdd n 1) m) (HAdd.hAdd (HAdd.hAdd n m) 1)
    k : Nat
    b : α
    t : Vector3 α k
    IH : ∀ (x : Eq (HAdd.hAdd (HAdd.hAdd n 1) k) (HAdd.hAdd (HAdd.hAdd n k) 1)), E …
    x✝ : Eq (HAdd.hAdd (HAdd.hAdd n 1) (HAdd.hAdd k 1)) (HAdd.hAdd (HAdd.hAdd n (H …
    e' : Eq (HAdd.hAdd (HAdd.hAdd n 1) k) (HAdd.hAdd (HAdd.hAdd n k) 1)
    ⊢ Eq (Vector3.insert a (Vector3.cons b (t.append v)) (Eq.recOn e' (i.add k)).f …
  -/
  simpa [IH] using Eq.recOn e' rfl
  /-
    🎉 no goals
  -/


/-- "Curried" exists, i.e. `∃ x₁ ... xₙ, f [x₁, ..., xₙ]`. -/
def VectorEx : ∀ k, (Vector3 α k → Prop) → Prop
  | 0, f => f []
  | succ k, f => ∃ x : α, VectorEx k fun v => f (x :: v)


/-- "Curried" forall, i.e. `∀ x₁ ... xₙ, f [x₁, ..., xₙ]`. -/
def VectorAll : ∀ k, (Vector3 α k → Prop) → Prop
  | 0, f => f []
  | succ k, f => ∀ x : α, VectorAll k fun v => f (x :: v)


theorem exists_vector_zero (f : Vector3 α 0 → Prop) : Exists f ↔ f [] :=
                     /-
                       α : Type u_1
                       f : Vector3 α 0 → Prop
                       x✝ : Exists f
                       v : Vector3 α 0
                       fv : f v
                       ⊢ f Vector3.nil
                     -/
  ⟨fun ⟨v, fv⟩ => by rw [← eq_nil v]; exact fv, fun f0 => ⟨[], f0⟩⟩
                                      /-
                                        🎉 no goals
                                      -/


theorem exists_vector_succ (f : Vector3 α (succ n) → Prop) : Exists f ↔ ∃ x v, f (x :: v) :=
                            /-
                              α : Type u_1
                              n : Nat
                              f : Vector3 α n.succ → Prop
                              x✝ : Exists f
                              v : Vector3 α n.succ
                              fv : f v
                              ⊢ f (Vector3.cons (?m.22809 x✝ v fv) (?m.22810 x✝ v fv))
                            -/
  ⟨fun ⟨v, fv⟩ => ⟨_, _, by rw [cons_head_tail v]; exact fv⟩, fun ⟨_, _, fxv⟩ => ⟨_, fxv⟩⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem vectorEx_iff_exists : ∀ {n} (f : Vector3 α n → Prop), VectorEx n f ↔ Exists f
  | 0, f => (exists_vector_zero f).symm
  | succ _, f =>
    Iff.trans (exists_congr fun _ => vectorEx_iff_exists _) (exists_vector_succ f).symm


theorem vectorAll_iff_forall : ∀ {n} (f : Vector3 α n → Prop), VectorAll n f ↔ ∀ v, f v
  | 0, _ => ⟨fun f0 v => v.nilElim f0, fun al => al []⟩
  | succ _, f =>
    (forall_congr' fun x => vectorAll_iff_forall fun v => f (x :: v)).trans
      ⟨fun al v => v.consElim al, fun al x v => al (x :: v)⟩


/-- `VectorAllP p v` is equivalent to `∀ i, p (v i)`, but unfolds directly to a conjunction,
  i.e. `VectorAllP p [0, 1, 2] = p 0 ∧ p 1 ∧ p 2`. -/
def VectorAllP (p : α → Prop) (v : Vector3 α n) : Prop :=
  Vector3.recOn v True fun a v IH =>
    @Vector3.recOn _ (fun _ => Prop) _ v (p a) fun _ _ _ => p a ∧ IH


@[simp]
theorem vectorAllP_nil (p : α → Prop) : VectorAllP p [] = True :=
  rfl


@[simp]
theorem vectorAllP_singleton (p : α → Prop) (x : α) : VectorAllP p (cons x []) = p x :=
  rfl


@[simp]
theorem vectorAllP_cons (p : α → Prop) (x : α) (v : Vector3 α n) :
    VectorAllP p (x :: v) ↔ p x ∧ VectorAllP p v :=
  Vector3.recOn v (iff_of_eq (and_true _)).symm fun _ _ _ => Iff.rfl


theorem vectorAllP_iff_forall (p : α → Prop) (v : Vector3 α n) :
    VectorAllP p v ↔ ∀ i, p (v i) := by
  /-
    α : Type u_1
    n : Nat
    p : α → Prop
    v : Vector3 α n
    ⊢ Iff (VectorAllP p v) (∀ (i : Fin2 n), p (v i))
  -/
  refine v.recOn ?_ ?_
    /-
      case refine_1
      α : Type u_1
      n : Nat
      p : α → Prop
      v : Vector3 α n
      ⊢ Iff (VectorAllP p Vector3.nil) (∀ (i : Fin2 0), p (Vector3.nil i))
    -/
  · exact ⟨fun _ => Fin2.elim0, fun _ => trivial⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      n : Nat
      p : α → Prop
      v : Vector3 α n
      ⊢ ∀ {n : Nat} (a : α) (w : Vector3 α n), Iff (VectorAllP p w) (∀ (i : Fin2 n), …
    -/
  · simp only [vectorAllP_cons]
    refine fun {n} a v IH =>
      (and_congr_right fun _ => IH).trans
        ⟨fun ⟨pa, h⟩ i => by
          refine i.cases' ?_ ?_
          exacts [pa, h], fun h => ⟨?_, fun i => ?_⟩⟩
      /-
        case refine_2.refine_1
        α : Type u_1
        n✝ : Nat
        p : α → Prop
        v✝ : Vector3 α n✝
        n : Nat
        a : α
        v : Vector3 α n
        IH : Iff (VectorAllP p v) (∀ (i : Fin2 n), p (v i))
        h : ∀ (i : Fin2 (HAdd.hAdd n 1)), p (Vector3.cons a v i)
        ⊢ p a
      -/
    · simpa using h fz
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        n✝ : Nat
        p : α → Prop
        v✝ : Vector3 α n✝
        n : Nat
        a : α
        v : Vector3 α n
        IH : Iff (VectorAllP p v) (∀ (i : Fin2 n), p (v i))
        h : ∀ (i : Fin2 (HAdd.hAdd n 1)), p (Vector3.cons a v i)
        i : Fin2 n
        ⊢ p (v i)
      -/
    · simpa using h (fs i)
      /-
        🎉 no goals
      -/


theorem VectorAllP.imp {p q : α → Prop} (h : ∀ x, p x → q x) {v : Vector3 α n}
    (al : VectorAllP p v) : VectorAllP q v :=
  (vectorAllP_iff_forall _ _).2 fun _ => h _ <| (vectorAllP_iff_forall _ _).1 al _


