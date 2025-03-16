/-- A walk is a sequence of adjacent vertices.  For vertices `u v : V`,
the type `walk u v` consists of all walks starting at `u` and ending at `v`.

We say that a walk *visits* the vertices it contains.  The set of vertices a
walk visits is `SimpleGraph.Walk.support`.

See `SimpleGraph.Walk.nil'` and `SimpleGraph.Walk.cons'` for patterns that
can be useful in definitions since they make the vertices explicit. -/
inductive Walk : V → V → Type u
  | nil {u : V} : Walk u u
  | cons {u v w : V} (h : G.Adj u v) (p : Walk v w) : Walk u w
  deriving DecidableEq


@[simps]
instance Walk.instInhabited (v : V) : Inhabited (G.Walk v v) := ⟨Walk.nil⟩


/-- The one-edge walk associated to a pair of adjacent vertices. -/
@[match_pattern, reducible]
def Adj.toWalk {G : SimpleGraph V} {u v : V} (h : G.Adj u v) : G.Walk u v :=
  Walk.cons h Walk.nil


/-- Pattern to get `Walk.nil` with the vertex as an explicit argument. -/
@[match_pattern]
abbrev nil' (u : V) : G.Walk u u := Walk.nil


/-- Pattern to get `Walk.cons` with the vertices as explicit arguments. -/
@[match_pattern]
abbrev cons' (u v w : V) (h : G.Adj u v) (p : G.Walk v w) : G.Walk u w := Walk.cons h p


/-- Change the endpoints of a walk using equalities. This is helpful for relaxing
definitional equality constraints and to be able to state otherwise difficult-to-state
lemmas. While this is a simple wrapper around `Eq.rec`, it gives a canonical way to write it.

The simp-normal form is for the `copy` to be pushed outward. That way calculations can
occur within the "copy context." -/
protected def copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') : G.Walk u' v' :=
  hu ▸ hv ▸ p


@[simp]
theorem copy_rfl_rfl {u v} (p : G.Walk u v) : p.copy rfl rfl = p := rfl


@[simp]
theorem copy_copy {u v u' v' u'' v''} (p : G.Walk u v)
    (hu : u = u') (hv : v = v') (hu' : u' = u'') (hv' : v' = v'') :
    (p.copy hu hv).copy hu' hv' = p.copy (hu.trans hu') (hv.trans hv') := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' u'' v'' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    hu' : Eq u' u''
    hv' : Eq v' v''
    ⊢ Eq ((p.copy hu hv).copy hu' hv') (p.copy ⋯ ⋯)
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u'' v'' : V
    p : G.Walk u'' v''
    ⊢ Eq ((p.copy ⋯ ⋯).copy ⋯ ⋯) (p.copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem copy_nil {u u'} (hu : u = u') : (Walk.nil : G.Walk u u).copy hu hu = Walk.nil := by
  /-
    V : Type u
    G : SimpleGraph V
    u u' : V
    hu : Eq u u'
    ⊢ Eq (SimpleGraph.Walk.nil.copy hu hu) SimpleGraph.Walk.nil
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' : V
    ⊢ Eq (SimpleGraph.Walk.nil.copy ⋯ ⋯) SimpleGraph.Walk.nil
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem copy_cons {u v w u' w'} (h : G.Adj u v) (p : G.Walk v w) (hu : u = u') (hw : w = w') :
    (Walk.cons h p).copy hu hw = Walk.cons (hu ▸ h) (p.copy rfl hw) := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w u' w' : V
    h : G.Adj u v
    p : G.Walk v w
    hu : Eq u u'
    hw : Eq w w'
    ⊢ Eq ((SimpleGraph.Walk.cons h p).copy hu hw) (SimpleGraph.Walk.cons ⋯ (p.copy …
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    v u' w' : V
    h : G.Adj u' v
    p : G.Walk v w'
    ⊢ Eq ((SimpleGraph.Walk.cons h p).copy ⋯ ⋯) (SimpleGraph.Walk.cons ⋯ (p.copy ⋯ …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem cons_copy {u v w v' w'} (h : G.Adj u v) (p : G.Walk v' w') (hv : v' = v) (hw : w' = w) :
    Walk.cons h (p.copy hv hw) = (Walk.cons (hv ▸ h) p).copy rfl hw := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w v' w' : V
    h : G.Adj u v
    p : G.Walk v' w'
    hv : Eq v' v
    hw : Eq w' w
    ⊢ Eq (SimpleGraph.Walk.cons h (p.copy hv hw)) ((SimpleGraph.Walk.cons ⋯ p).cop …
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u v' w' : V
    p : G.Walk v' w'
    h : G.Adj u v'
    ⊢ Eq (SimpleGraph.Walk.cons h (p.copy ⋯ ⋯)) ((SimpleGraph.Walk.cons ⋯ p).copy  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem exists_eq_cons_of_ne {u v : V} (hne : u ≠ v) :
    ∀ (p : G.Walk u v), ∃ (w : V) (h : G.Adj u w) (p' : G.Walk w v), p = cons h p'
  | nil => (hne rfl).elim
  | cons h p' => ⟨_, h, p', rfl⟩


/-- The length of a walk is the number of edges/darts along it. -/
def length {u v : V} : G.Walk u v → ℕ
  | nil => 0
  | cons _ q => q.length.succ


/-- The concatenation of two compatible walks. -/
@[trans]
def append {u v w : V} : G.Walk u v → G.Walk v w → G.Walk u w
  | nil, q => q
  | cons h p, q => cons h (p.append q)


/-- The reversed version of `SimpleGraph.Walk.cons`, concatenating an edge to
the end of a walk. -/
def concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) : G.Walk u w := p.append (cons h nil)


theorem concat_eq_append {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
    p.concat h = p.append (cons h nil) := rfl


/-- The concatenation of the reverse of the first walk with the second walk. -/
protected def reverseAux {u v w : V} : G.Walk u v → G.Walk u w → G.Walk v w
  | nil, q => q
  | cons h p, q => Walk.reverseAux p (cons (G.symm h) q)


/-- The walk in reverse. -/
@[symm]
def reverse {u v : V} (w : G.Walk u v) : G.Walk v u := w.reverseAux nil


/-- Get the `n`th vertex from a walk, where `n` is generally expected to be
between `0` and `p.length`, inclusive.
If `n` is greater than or equal to `p.length`, the result is the path's endpoint. -/
def getVert {u v : V} : G.Walk u v → ℕ → V
  | nil, _ => u
  | cons _ _, 0 => u
  | cons _ q, n + 1 => q.getVert n


@[simp]
                                                                    /-
                                                                      V : Type u
                                                                      G : SimpleGraph V
                                                                      u v : V
                                                                      w : G.Walk u v
                                                                      ⊢ Eq (w.getVert 0) u
                                                                    -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
theorem getVert_zero {u v} (w : G.Walk u v) : w.getVert 0 = u := by cases w <;> rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem getVert_of_length_le {u v} (w : G.Walk u v) {i : ℕ} (hi : w.length ≤ i) :
    w.getVert i = v := by
  induction w generalizing i with
  | nil => rfl
  | cons _ _ ih =>
    cases i
    · cases hi
    · exact ih (Nat.succ_le_succ_iff.1 hi)


@[simp]
theorem getVert_length {u v} (w : G.Walk u v) : w.getVert w.length = v :=
  w.getVert_of_length_le rfl.le


theorem adj_getVert_succ {u v} (w : G.Walk u v) {i : ℕ} (hi : i < w.length) :
    G.Adj (w.getVert i) (w.getVert (i + 1)) := by
  induction w generalizing i with
  | nil => cases hi
  | cons hxy _ ih =>
    cases i
    · simp [getVert, hxy]
    · exact ih (Nat.succ_lt_succ_iff.1 hi)


lemma getVert_cons_one {u v w} (q : G.Walk v w) (hadj : G.Adj u v) :
    (q.cons hadj).getVert 1 = v := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    q : G.Walk v w
    hadj : G.Adj u v
    ⊢ Eq ((SimpleGraph.Walk.cons hadj q).getVert 1) v
  -/
  have : (q.cons hadj).getVert 1 = q.getVert 0 := rfl
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    q : G.Walk v w
    hadj : G.Adj u v
    this : Eq ((SimpleGraph.Walk.cons hadj q).getVert 1) (q.getVert 0)
    ⊢ Eq ((SimpleGraph.Walk.cons hadj q).getVert 1) v
  -/
  simpa [getVert_zero] using this
  /-
    🎉 no goals
  -/


@[simp]
lemma getVert_cons_succ {u v w n} (p : G.Walk v w) (h : G.Adj u v) :
    (p.cons h).getVert (n + 1) = p.getVert n := rfl


lemma getVert_cons {u v w n} (p : G.Walk v w) (h : G.Adj u v) (hn : n ≠ 0) :
    (p.cons h).getVert n = p.getVert (n - 1) := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    n : Nat
    p : G.Walk v w
    h : G.Adj u v
    hn : Ne n 0
    ⊢ Eq ((SimpleGraph.Walk.cons h p).getVert n) (p.getVert (HSub.hSub n 1))
  -/
  obtain ⟨n, rfl⟩ := Nat.exists_eq_add_one_of_ne_zero hn
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk v w
    h : G.Adj u v
    n : Nat
    hn : Ne (HAdd.hAdd n 1) 0
    ⊢ Eq ((SimpleGraph.Walk.cons h p).getVert (HAdd.hAdd n 1)) (p.getVert (HSub.hS …
  -/
  rw [getVert_cons_succ, Nat.add_sub_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem cons_append {u v w x : V} (h : G.Adj u v) (p : G.Walk v w) (q : G.Walk w x) :
    (cons h p).append q = cons h (p.append q) := rfl


@[simp]
theorem cons_nil_append {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    (cons h nil).append p = cons h p := rfl


@[simp]
theorem nil_append {u v : V} (p : G.Walk u v) : nil.append p = p :=
  rfl


@[simp]
theorem append_nil {u v : V} (p : G.Walk u v) : p.append nil = p := by
  induction p with
  | nil => rw [nil_append]
  | cons _ _ ih => rw [cons_append, ih]


theorem append_assoc {u v w x : V} (p : G.Walk u v) (q : G.Walk v w) (r : G.Walk w x) :
    p.append (q.append r) = (p.append q).append r := by
  induction p with
  | nil => rw [nil_append, nil_append]
  | cons h p' ih => rw [cons_append, cons_append, cons_append, ih]


@[simp]
theorem append_copy_copy {u v w u' v' w'} (p : G.Walk u v) (q : G.Walk v w)
    (hu : u = u') (hv : v = v') (hw : w = w') :
    (p.copy hu hv).append (q.copy hv hw) = (p.append q).copy hu hw := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w u' v' w' : V
    p : G.Walk u v
    q : G.Walk v w
    hu : Eq u u'
    hv : Eq v v'
    hw : Eq w w'
    ⊢ Eq ((p.copy hu hv).append (q.copy hv hw)) ((p.append q).copy hu hw)
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' w' : V
    p : G.Walk u' v'
    q : G.Walk v' w'
    ⊢ Eq ((p.copy ⋯ ⋯).append (q.copy ⋯ ⋯)) ((p.append q).copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem concat_nil {u v : V} (h : G.Adj u v) : nil.concat h = cons h nil := rfl


@[simp]
theorem concat_cons {u v w x : V} (h : G.Adj u v) (p : G.Walk v w) (h' : G.Adj w x) :
    (cons h p).concat h' = cons h (p.concat h') := rfl


theorem append_concat {u v w x : V} (p : G.Walk u v) (q : G.Walk v w) (h : G.Adj w x) :
    p.append (q.concat h) = (p.append q).concat h := append_assoc _ _ _


theorem concat_append {u v w x : V} (p : G.Walk u v) (h : G.Adj v w) (q : G.Walk w x) :
    (p.concat h).append q = p.append (cons h q) := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w x : V
    p : G.Walk u v
    h : G.Adj v w
    q : G.Walk w x
    ⊢ Eq ((p.concat h).append q) (p.append (SimpleGraph.Walk.cons h q))
  -/
  rw [concat_eq_append, ← append_assoc, cons_nil_append]
  /-
    🎉 no goals
  -/


/-- A non-trivial `cons` walk is representable as a `concat` walk. -/
theorem exists_cons_eq_concat {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    ∃ (x : V) (q : G.Walk u x) (h' : G.Adj x w), cons h p = q.concat h' := by
  induction p generalizing u with
  | nil => exact ⟨_, nil, h, rfl⟩
  | cons h' p ih =>
    obtain ⟨y, q, h'', hc⟩ := ih h'
    refine ⟨y, cons h q, h'', ?_⟩
    rw [concat_cons, hc]


/-- A non-trivial `concat` walk is representable as a `cons` walk. -/
theorem exists_concat_eq_cons {u v w : V} :
    ∀ (p : G.Walk u v) (h : G.Adj v w),
      ∃ (x : V) (h' : G.Adj u x) (q : G.Walk x w), p.concat h = cons h' q
  | nil, h => ⟨_, h, nil, rfl⟩
  | cons h' p, h => ⟨_, h', Walk.concat p h, concat_cons _ _ _⟩


@[simp]
theorem reverse_nil {u : V} : (nil : G.Walk u u).reverse = nil := rfl


theorem reverse_singleton {u v : V} (h : G.Adj u v) : (cons h nil).reverse = cons (G.symm h) nil :=
  rfl


@[simp]
theorem cons_reverseAux {u v w x : V} (p : G.Walk u v) (q : G.Walk w x) (h : G.Adj w u) :
    (cons h p).reverseAux q = p.reverseAux (cons (G.symm h) q) := rfl


@[simp]
protected theorem append_reverseAux {u v w x : V}
    (p : G.Walk u v) (q : G.Walk v w) (r : G.Walk u x) :
    (p.append q).reverseAux r = q.reverseAux (p.reverseAux r) := by
  induction p with
  | nil => rfl
  | cons h _ ih => exact ih q (cons (G.symm h) r)


@[simp]
protected theorem reverseAux_append {u v w x : V}
    (p : G.Walk u v) (q : G.Walk u w) (r : G.Walk w x) :
    (p.reverseAux q).append r = p.reverseAux (q.append r) := by
  induction p with
  | nil => rfl
  | cons h _ ih => simp [ih (cons (G.symm h) q)]


protected theorem reverseAux_eq_reverse_append {u v w : V} (p : G.Walk u v) (q : G.Walk u w) :
                                              /-
                                                V : Type u
                                                G : SimpleGraph V
                                                u v w : V
                                                p : G.Walk u v
                                                q : G.Walk u w
                                                ⊢ Eq (p.reverseAux q) (p.reverse.append q)
                                              -/
    p.reverseAux q = p.reverse.append q := by simp [reverse]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem reverse_cons {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
                                                                      /-
                                                                        V : Type u
                                                                        G : SimpleGraph V
                                                                        u v w : V
                                                                        h : G.Adj u v
                                                                        p : G.Walk v w
                                                                        ⊢ Eq (SimpleGraph.Walk.cons h p).reverse (p.reverse.append (SimpleGraph.Walk.c …
                                                                      -/
    (cons h p).reverse = p.reverse.append (cons (G.symm h) nil) := by simp [reverse]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem reverse_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).reverse = p.reverse.copy hv hu := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (p.copy hu hv).reverse (p.reverse.copy hv hu)
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (p.copy ⋯ ⋯).reverse (p.reverse.copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem reverse_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
                                                            /-
                                                              V : Type u
                                                              G : SimpleGraph V
                                                              u v w : V
                                                              p : G.Walk u v
                                                              q : G.Walk v w
                                                              ⊢ Eq (p.append q).reverse (q.reverse.append p.reverse)
                                                            -/
    (p.append q).reverse = q.reverse.append p.reverse := by simp [reverse]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem reverse_concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
                                                           /-
                                                             V : Type u
                                                             G : SimpleGraph V
                                                             u v w : V
                                                             p : G.Walk u v
                                                             h : G.Adj v w
                                                             ⊢ Eq (p.concat h).reverse (SimpleGraph.Walk.cons ⋯ p.reverse)
                                                           -/
    (p.concat h).reverse = cons (G.symm h) p.reverse := by simp [concat_eq_append]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem reverse_reverse {u v : V} (p : G.Walk u v) : p.reverse.reverse = p := by
  induction p with
  | nil => rfl
  | cons _ _ ih => simp [ih]


theorem reverse_surjective {u v : V} : Function.Surjective (reverse : G.Walk u v → _) :=
  RightInverse.surjective reverse_reverse


theorem reverse_injective {u v : V} : Function.Injective (reverse : G.Walk u v → _) :=
  RightInverse.injective reverse_reverse


theorem reverse_bijective {u v : V} : Function.Bijective (reverse : G.Walk u v → _) :=
  And.intro reverse_injective reverse_surjective


@[simp]
theorem length_nil {u : V} : (nil : G.Walk u u).length = 0 := rfl


@[simp]
theorem length_cons {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    (cons h p).length = p.length + 1 := rfl


@[simp]
theorem length_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).length = p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (p.copy hu hv).length p.length
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (p.copy ⋯ ⋯).length p.length
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem length_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
    (p.append q).length = p.length + q.length := by
  induction p with
  | nil => simp
  | cons _ _ ih => simp [ih, add_comm, add_left_comm, add_assoc]


@[simp]
theorem length_concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
    (p.concat h).length = p.length + 1 := length_append _ _


@[simp]
protected theorem length_reverseAux {u v w : V} (p : G.Walk u v) (q : G.Walk u w) :
    (p.reverseAux q).length = p.length + q.length := by
  induction p with
  | nil => simp!
  | cons _ _ ih => simp [ih, Nat.succ_add, Nat.add_assoc]


@[simp]
                                                                                      /-
                                                                                        V : Type u
                                                                                        G : SimpleGraph V
                                                                                        u v : V
                                                                                        p : G.Walk u v
                                                                                        ⊢ Eq p.reverse.length p.length
                                                                                      -/
theorem length_reverse {u v : V} (p : G.Walk u v) : p.reverse.length = p.length := by simp [reverse]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


theorem eq_of_length_eq_zero {u v : V} : ∀ {p : G.Walk u v}, p.length = 0 → u = v
  | nil, _ => rfl


theorem adj_of_length_eq_one {u v : V} : ∀ {p : G.Walk u v}, p.length = 1 → G.Adj u v
  | cons h nil, _ => h


@[simp]
theorem exists_length_eq_zero_iff {u v : V} : (∃ p : G.Walk u v, p.length = 0) ↔ u = v := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    ⊢ Iff (Exists fun p => Eq p.length 0) (Eq u v)
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      u v : V
      ⊢ (Exists fun p => Eq p.length 0) → Eq u v
    -/
  · rintro ⟨p, hp⟩
    /-
      case mp.intro
      V : Type u
      G : SimpleGraph V
      u v : V
      p : G.Walk u v
      hp : Eq p.length 0
      ⊢ Eq u v
    -/
    exact eq_of_length_eq_zero hp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      u v : V
      ⊢ Eq u v → Exists fun p => Eq p.length 0
    -/
  · rintro rfl
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      u : V
      ⊢ Exists fun p => Eq p.length 0
    -/
    exact ⟨nil, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
                                                                                   /-
                                                                                     V : Type u
                                                                                     G : SimpleGraph V
                                                                                     u : V
                                                                                     p : G.Walk u u
                                                                                     ⊢ Iff (Eq p.length 0) (Eq p SimpleGraph.Walk.nil)
                                                                                   -/
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
theorem length_eq_zero_iff {u : V} {p : G.Walk u u} : p.length = 0 ↔ p = nil := by cases p <;> simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


theorem getVert_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) (i : ℕ) :
    (p.append q).getVert i = if i < p.length then p.getVert i else q.getVert (i - p.length) := by
  induction p generalizing i with
  | nil => simp
  | cons h p ih => cases i <;> simp [getVert, ih, Nat.succ_lt_succ_iff]


theorem getVert_reverse {u v : V} (p : G.Walk u v) (i : ℕ) :
    p.reverse.getVert i = p.getVert (p.length - i) := by
  induction p with
  | nil => rfl
  | cons h p ih =>
    simp only [reverse_cons, getVert_append, length_reverse, ih, length_cons]
    split_ifs
    next hi =>
      rw [Nat.succ_sub hi.le]
      simp [getVert]
    next hi =>
      obtain rfl | hi' := Nat.eq_or_lt_of_not_lt hi
      · simp [getVert]
      · rw [Nat.eq_add_of_sub_eq (Nat.sub_pos_of_lt hi') rfl, Nat.sub_eq_zero_of_le hi']
        simp [getVert]


/-- Auxiliary definition for `SimpleGraph.Walk.concatRec` -/
def concatRecAux {u v : V} : (p : G.Walk u v) → motive v u p.reverse
  | nil => Hnil
  | cons h p => reverse_cons h p ▸ Hconcat p.reverse h.symm (concatRecAux p)


/-- Recursor on walks by inducting on `SimpleGraph.Walk.concat`.

This is inducting from the opposite end of the walk compared
to `SimpleGraph.Walk.rec`, which inducts on `SimpleGraph.Walk.cons`. -/
@[elab_as_elim]
def concatRec {u v : V} (p : G.Walk u v) : motive u v p :=
  reverse_reverse p ▸ concatRecAux @Hnil @Hconcat p.reverse


@[simp]
theorem concatRec_nil (u : V) :
    @concatRec _ _ motive @Hnil @Hconcat _ _ (nil : G.Walk u u) = Hnil := rfl


@[simp]
theorem concatRec_concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
    @concatRec _ _ motive @Hnil @Hconcat _ _ (p.concat h) =
      Hconcat p h (concatRec @Hnil @Hconcat p) := by
  /-
    V : Type u
    G : SimpleGraph V
    motive : (u v : V) → G.Walk u v → Sort u_1
    Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
    Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
    u v w : V
    p : G.Walk u v
    h : G.Adj v w
    ⊢ Eq (SimpleGraph.Walk.concatRec Hnil Hconcat (p.concat h)) (Hconcat p h (Simp …
  -/
  simp only [concatRec]
  /-
    V : Type u
    G : SimpleGraph V
    motive : (u v : V) → G.Walk u v → Sort u_1
    Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
    Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
    u v w : V
    p : G.Walk u v
    h : G.Adj v w
    ⊢ Eq (Eq.rec (SimpleGraph.Walk.concatRecAux Hnil Hconcat (p.concat h).reverse) …
  -/
  apply eq_of_heq
  /-
    case h
    V : Type u
    G : SimpleGraph V
    motive : (u v : V) → G.Walk u v → Sort u_1
    Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
    Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
    u v w : V
    p : G.Walk u v
    h : G.Adj v w
    ⊢ HEq (Eq.rec (SimpleGraph.Walk.concatRecAux Hnil Hconcat (p.concat h).reverse …
  -/
  apply rec_heq_of_heq
  /-
    case h.h
    V : Type u
    G : SimpleGraph V
    motive : (u v : V) → G.Walk u v → Sort u_1
    Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
    Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
    u v w : V
    p : G.Walk u v
    h : G.Adj v w
    ⊢ HEq (SimpleGraph.Walk.concatRecAux Hnil Hconcat (p.concat h).reverse) (Hconc …
  -/
  trans concatRecAux @Hnil @Hconcat (cons h.symm p.reverse)
    /-
      V : Type u
      G : SimpleGraph V
      motive : (u v : V) → G.Walk u v → Sort u_1
      Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
      Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
      u v w : V
      p : G.Walk u v
      h : G.Adj v w
      ⊢ HEq (SimpleGraph.Walk.concatRecAux Hnil Hconcat (p.concat h).reverse) (Simpl …
    -/
  · congr
    /-
      case e_8.h
      V : Type u
      G : SimpleGraph V
      motive : (u v : V) → G.Walk u v → Sort u_1
      Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
      Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
      u v w : V
      p : G.Walk u v
      h : G.Adj v w
      ⊢ Eq (p.concat h).reverse (SimpleGraph.Walk.cons ⋯ p.reverse)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      V : Type u
      G : SimpleGraph V
      motive : (u v : V) → G.Walk u v → Sort u_1
      Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
      Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
      u v w : V
      p : G.Walk u v
      h : G.Adj v w
      ⊢ HEq (SimpleGraph.Walk.concatRecAux Hnil Hconcat (SimpleGraph.Walk.cons ⋯ p.r …
    -/
  · rw [concatRecAux, rec_heq_iff_heq]
    /-
      V : Type u
      G : SimpleGraph V
      motive : (u v : V) → G.Walk u v → Sort u_1
      Hnil : {u : V} → motive u u SimpleGraph.Walk.nil
      Hconcat : {u v w : V} → (p : G.Walk u v) → (h : G.Adj v w) → motive u v p → mo …
      u v w : V
      p : G.Walk u v
      h : G.Adj v w
      ⊢ HEq (Hconcat p.reverse.reverse ⋯ (SimpleGraph.Walk.concatRecAux Hnil Hconcat …
    -/
              /-
                🎉 no goals
              -/
    congr <;> simp [heq_rec_iff_heq]
              /-
                🎉 no goals
              -/


theorem concat_ne_nil {u v : V} (p : G.Walk u v) (h : G.Adj v u) : p.concat h ≠ nil := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    h : G.Adj v u
    ⊢ Ne (p.concat h) SimpleGraph.Walk.nil
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp [concat]
              /-
                🎉 no goals
              -/


theorem concat_inj {u v v' w : V} {p : G.Walk u v} {h : G.Adj v w} {p' : G.Walk u v'}
    {h' : G.Adj v' w} (he : p.concat h = p'.concat h') : ∃ hv : v = v', p.copy rfl hv = p' := by
  induction p with
  | nil =>
    cases p'
    · exact ⟨rfl, rfl⟩
    · exfalso
      simp only [concat_nil, concat_cons, cons.injEq] at he
      obtain ⟨rfl, he⟩ := he
      simp only [heq_iff_eq] at he
      exact concat_ne_nil _ _ he.symm
  | cons _ _ ih =>
    rw [concat_cons] at he
    cases p'
    · exfalso
      simp only [concat_nil, cons.injEq] at he
      obtain ⟨rfl, he⟩ := he
      rw [heq_iff_eq] at he
      exact concat_ne_nil _ _ he
    · rw [concat_cons, cons.injEq] at he
      obtain ⟨rfl, he⟩ := he
      rw [heq_iff_eq] at he
      obtain ⟨rfl, rfl⟩ := ih he
      exact ⟨rfl, rfl⟩


/-- The `support` of a walk is the list of vertices it visits in order. -/
def support {u v : V} : G.Walk u v → List V
  | nil => [u]
  | cons _ p => u :: p.support


/-- The `darts` of a walk is the list of darts it visits in order. -/
def darts {u v : V} : G.Walk u v → List G.Dart
  | nil => []
  | cons h p => ⟨(u, _), h⟩ :: p.darts


/-- The `edges` of a walk is the list of edges it visits in order.
This is defined to be the list of edges underlying `SimpleGraph.Walk.darts`. -/
def edges {u v : V} (p : G.Walk u v) : List (Sym2 V) := p.darts.map Dart.edge


@[simp]
theorem support_nil {u : V} : (nil : G.Walk u u).support = [u] := rfl


@[simp]
theorem support_cons {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    (cons h p).support = u :: p.support := rfl


@[simp]
theorem support_concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
    (p.concat h).support = p.support.concat w := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    h : G.Adj v w
    ⊢ Eq (p.concat h).support (p.support.concat w)
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*, concat_nil]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem support_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).support = p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (p.copy hu hv).support p.support
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (p.copy ⋯ ⋯).support p.support
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem support_append {u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    (p.append p').support = p.support ++ p'.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (p.append p').support (HAppend.hAppend p.support p'.support.tail)
  -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  induction p <;> cases p' <;> simp [*]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem support_reverse {u v : V} (p : G.Walk u v) : p.reverse.support = p.support.reverse := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq p.reverse.support p.support.reverse
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [support_append, *]
                  /-
                    🎉 no goals
                  -/


@[simp]
                                                                         /-
                                                                           V : Type u
                                                                           G : SimpleGraph V
                                                                           u v : V
                                                                           p : G.Walk u v
                                                                           ⊢ Ne p.support List.nil
                                                                         -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
theorem support_ne_nil {u v : V} (p : G.Walk u v) : p.support ≠ [] := by cases p <;> simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem head_support {G : SimpleGraph V} {a b : V} (p : G.Walk a b) :
                       /-
                         V : Type u
                         V' : Type v
                         V'' : Type w
                         G✝ : SimpleGraph V
                         G' : SimpleGraph V'
                         G'' : SimpleGraph V''
                         G : SimpleGraph V
                         a b : V
                         p : G.Walk a b
                         ⊢ Ne p.support List.nil
                       -/
                       /-
                         🎉 no goals
                       -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    p.support.head (by simp) = a := by cases p <;> simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem getLast_support {G : SimpleGraph V} {a b : V} (p : G.Walk a b) :
                          /-
                            V : Type u
                            V' : Type v
                            V'' : Type w
                            G✝ : SimpleGraph V
                            G' : SimpleGraph V'
                            G'' : SimpleGraph V''
                            G : SimpleGraph V
                            a b : V
                            p : G.Walk a b
                            ⊢ Ne p.support List.nil
                          -/
    p.support.getLast (by simp) = b := by
                          /-
                            🎉 no goals
                          -/
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    p : G.Walk a b
    ⊢ Eq (p.support.getLast ⋯) b
  -/
  induction p
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      a b u✝ : V
      ⊢ Eq (SimpleGraph.Walk.nil.support.getLast ⋯) u✝
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      a b u✝ v✝ w✝ : V
      h✝ : G.Adj u✝ v✝
      p✝ : G.Walk v✝ w✝
      p_ih✝ : Eq (p✝.support.getLast ⋯) w✝
      ⊢ Eq ((SimpleGraph.Walk.cons h✝ p✝).support.getLast ⋯) w✝
    -/
  · simpa
    /-
      🎉 no goals
    -/


theorem tail_support_append {u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    (p.append p').support.tail = p.support.tail ++ p'.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (p.append p').support.tail (HAppend.hAppend p.support.tail p'.support.tail)
  -/
  rw [support_append, List.tail_append_of_ne_nil (support_ne_nil _)]
  /-
    🎉 no goals
  -/


theorem support_eq_cons {u v : V} (p : G.Walk u v) : p.support = u :: p.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq p.support (List.cons u p.support.tail)
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp
              /-
                🎉 no goals
              -/


@[simp]
                                                                           /-
                                                                             V : Type u
                                                                             G : SimpleGraph V
                                                                             u v : V
                                                                             p : G.Walk u v
                                                                             ⊢ Membership.mem p.support u
                                                                           -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
theorem start_mem_support {u v : V} (p : G.Walk u v) : u ∈ p.support := by cases p <;> simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
                                                                         /-
                                                                           V : Type u
                                                                           G : SimpleGraph V
                                                                           u v : V
                                                                           p : G.Walk u v
                                                                           ⊢ Membership.mem p.support v
                                                                         -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
theorem end_mem_support {u v : V} (p : G.Walk u v) : v ∈ p.support := by induction p <;> simp [*]
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem support_nonempty {u v : V} (p : G.Walk u v) : { w | w ∈ p.support }.Nonempty :=
         /-
           V : Type u
           G : SimpleGraph V
           u v : V
           p : G.Walk u v
           ⊢ Membership.mem (setOf fun w => Membership.mem p.support w) u
         -/
  ⟨u, by simp⟩
         /-
           🎉 no goals
         -/


theorem mem_support_iff {u v w : V} (p : G.Walk u v) :
                                                     /-
                                                       V : Type u
                                                       G : SimpleGraph V
                                                       u v w : V
                                                       p : G.Walk u v
                                                       ⊢ Iff (Membership.mem p.support w) (Or (Eq w u) (Membership.mem p.support.tail …
                                                     -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    w ∈ p.support ↔ w = u ∨ w ∈ p.support.tail := by cases p <;> simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                                     /-
                                                                                       V : Type u
                                                                                       G : SimpleGraph V
                                                                                       u v : V
                                                                                       ⊢ Iff (Membership.mem SimpleGraph.Walk.nil.support u) (Eq u v)
                                                                                     -/
theorem mem_support_nil_iff {u v : V} : u ∈ (nil : G.Walk v v).support ↔ u = v := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem mem_tail_support_append_iff {t u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    t ∈ (p.append p').support.tail ↔ t ∈ p.support.tail ∨ t ∈ p'.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Iff (Membership.mem (p.append p').support.tail t) (Or (Membership.mem p.supp …
  -/
  rw [tail_support_append, List.mem_append]
  /-
    🎉 no goals
  -/


@[simp]
theorem end_mem_tail_support_of_ne {u v : V} (h : u ≠ v) (p : G.Walk u v) : v ∈ p.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    h : Ne u v
    p : G.Walk u v
    ⊢ Membership.mem p.support.tail v
  -/
  obtain ⟨_, _, _, rfl⟩ := exists_eq_cons_of_ne h p
  /-
    case intro.intro.intro
    V : Type u
    G : SimpleGraph V
    u v : V
    h : Ne u v
    w✝² : V
    w✝¹ : G.Adj u w✝²
    w✝ : G.Walk w✝² v
    ⊢ Membership.mem (SimpleGraph.Walk.cons w✝¹ w✝).support.tail v
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, nolint unusedHavesSuffices]
theorem mem_support_append_iff {t u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    t ∈ (p.append p').support ↔ t ∈ p.support ∨ t ∈ p'.support := by
  /-
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Iff (Membership.mem (p.append p').support t) (Or (Membership.mem p.support t …
  -/
  simp only [mem_support_iff, mem_tail_support_append_iff]
  /-
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Iff (Or (Eq t u) (Or (Membership.mem p.support.tail t) (Membership.mem p'.su …
  -/
  obtain rfl | h := eq_or_ne t v <;> obtain rfl | h' := eq_or_ne t u <;>
    -- this `have` triggers the unusedHavesSuffices linter:
     /-
       case inl.inl
       V : Type u
       G : SimpleGraph V
       t w : V
       p' : G.Walk t w
       p : G.Walk t t
       ⊢ Iff (Or (Eq t t) (Or (Membership.mem p.support.tail t) (Membership.mem p'.su …
     -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
    (try have := h'.symm) <;> simp [*]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem subset_support_append_left {V : Type u} {G : SimpleGraph V} {u v w : V}
    (p : G.Walk u v) (q : G.Walk v w) : p.support ⊆ (p.append q).support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    ⊢ HasSubset.Subset p.support (p.append q).support
  -/
  simp only [Walk.support_append, List.subset_append_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem subset_support_append_right {V : Type u} {G : SimpleGraph V} {u v w : V}
    (p : G.Walk u v) (q : G.Walk v w) : q.support ⊆ (p.append q).support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    ⊢ HasSubset.Subset q.support (p.append q).support
  -/
  intro h
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    q : G.Walk v w
    h : V
    ⊢ Membership.mem q.support h → Membership.mem (p.append q).support h
  -/
  simp +contextual only [mem_support_append_iff, or_true, imp_true_iff]
  /-
    🎉 no goals
  -/


lemma getVert_eq_support_get? {u v n} (p : G.Walk u v) (h2 : n ≤ p.length) :
    p.getVert n = p.support[n]? := by
  match p with
  | .nil => simp_all
  | .cons h q =>
    simp only [Walk.support_cons]
    by_cases hn : n = 0
    · simp only [hn, getVert_zero, List.length_cons, Nat.zero_lt_succ, List.getElem?_eq_getElem,
      List.getElem_cons_zero]
    · push_neg at hn
      nth_rewrite 2 [← Nat.sub_one_add_one hn]
      rw [Walk.getVert_cons q h hn, List.getElem?_cons_succ]
      exact getVert_eq_support_get? q (Nat.sub_le_of_le_add (Walk.length_cons _ _ ▸ h2))


theorem coe_support {u v : V} (p : G.Walk u v) :
                                                          /-
                                                            V : Type u
                                                            G : SimpleGraph V
                                                            u v : V
                                                            p : G.Walk u v
                                                            ⊢ Eq (↑p.support) (HAdd.hAdd (Singleton.singleton u) ↑p.support.tail)
                                                          -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (p.support : Multiset V) = {u} + p.support.tail := by cases p <;> rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem coe_support_append {u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    ((p.append p').support : Multiset V) = {u} + p.support.tail + p'.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (↑(p.append p').support) (HAdd.hAdd (HAdd.hAdd (Singleton.singleton u) ↑p …
  -/
  rw [support_append, ← Multiset.coe_add, coe_support]
  /-
    🎉 no goals
  -/


theorem coe_support_append' [DecidableEq V] {u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    ((p.append p').support : Multiset V) = p.support + p'.support - {v} := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (↑(p.append p').support) (HSub.hSub (HAdd.hAdd ↑p.support ↑p'.support) (S …
  -/
  rw [support_append, ← Multiset.coe_add]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (HAdd.hAdd ↑p.support ↑p'.support.tail) (HSub.hSub (HAdd.hAdd ↑p.support  …
  -/
  simp only [coe_support]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Singleton.singleton u) ↑p.support.tail) ↑p'.suppor …
  -/
  rw [add_comm ({v} : Multiset V)]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Singleton.singleton u) ↑p.support.tail) ↑p'.suppor …
  -/
  simp only [← add_assoc, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


theorem chain_adj_support {u v w : V} (h : G.Adj u v) :
    ∀ (p : G.Walk v w), List.Chain G.Adj u p.support
  | nil => List.Chain.cons h List.Chain.nil
  | cons h' p => List.Chain.cons h (chain_adj_support h' p)


theorem chain'_adj_support {u v : V} : ∀ (p : G.Walk u v), List.Chain' G.Adj p.support
  | nil => List.Chain.nil
  | cons h p => chain_adj_support h p


theorem chain_dartAdj_darts {d : G.Dart} {v w : V} (h : d.snd = v) (p : G.Walk v w) :
    List.Chain G.DartAdj d p.darts := by
  induction p generalizing d with
  | nil => exact List.Chain.nil
  -- Porting note: needed to defer `h` and `rfl` to help elaboration
  | cons h' p ih => exact List.Chain.cons (by exact h) (ih (by rfl))


theorem chain'_dartAdj_darts {u v : V} : ∀ (p : G.Walk u v), List.Chain' G.DartAdj p.darts
  | nil => trivial
  -- Porting note: needed to defer `rfl` to help elaboration
                                        /-
                                          V : Type u
                                          G : SimpleGraph V
                                          u v✝¹ v v✝ : V
                                          h : G.Adj u v✝
                                          p : G.Walk v✝ v
                                          ⊢ Eq { fst := u, snd := v✝, adj := h }.toProd.2 v✝
                                        -/
  | cons h p => chain_dartAdj_darts (by rfl) p
                                        /-
                                          🎉 no goals
                                        -/


/-- Every edge in a walk's edge list is an edge of the graph.
It is written in this form (rather than using `⊆`) to avoid unsightly coercions. -/
theorem edges_subset_edgeSet {u v : V} :
    ∀ (p : G.Walk u v) ⦃e : Sym2 V⦄, e ∈ p.edges → e ∈ G.edgeSet
  | cons h' p', e, h => by
    /-
      V : Type u
      G : SimpleGraph V
      u v v✝ : V
      h' : G.Adj u v✝
      p' : G.Walk v✝ v
      e : Sym2 V
      h : Membership.mem (SimpleGraph.Walk.cons h' p').edges e
      ⊢ Membership.mem G.edgeSet e
    -/
    cases h
      /-
        case head
        V : Type u
        G : SimpleGraph V
        u v v✝ : V
        h' : G.Adj u v✝
        p' : G.Walk v✝ v
        ⊢ Membership.mem G.edgeSet { fst := u, snd := v✝, adj := h' }.edge
      -/
    · exact h'
      /-
        🎉 no goals
      -/
    /-
      case tail
      V : Type u
      G : SimpleGraph V
      u v v✝ : V
      h' : G.Adj u v✝
      p' : G.Walk v✝ v
      e : Sym2 V
      a✝ : List.Mem e (List.map SimpleGraph.Dart.edge p'.darts)
      ⊢ Membership.mem G.edgeSet e
    -/
    next h' => exact edges_subset_edgeSet p' h'
    /-
      🎉 no goals
    -/


theorem adj_of_mem_edges {u v x y : V} (p : G.Walk u v) (h : s(x, y) ∈ p.edges) : G.Adj x y :=
  edges_subset_edgeSet p h


@[simp]
theorem darts_nil {u : V} : (nil : G.Walk u u).darts = [] := rfl


@[simp]
theorem darts_cons {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    (cons h p).darts = ⟨(u, v), h⟩ :: p.darts := rfl


@[simp]
theorem darts_concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
    (p.concat h).darts = p.darts.concat ⟨(v, w), h⟩ := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    h : G.Adj v w
    ⊢ Eq (p.concat h).darts (p.darts.concat { fst := v, snd := w, adj := h })
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*, concat_nil]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem darts_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).darts = p.darts := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (p.copy hu hv).darts p.darts
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (p.copy ⋯ ⋯).darts p.darts
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem darts_append {u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
    (p.append p').darts = p.darts ++ p'.darts := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk u v
    p' : G.Walk v w
    ⊢ Eq (p.append p').darts (HAppend.hAppend p.darts p'.darts)
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem darts_reverse {u v : V} (p : G.Walk u v) :
    p.reverse.darts = (p.darts.map Dart.symm).reverse := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq p.reverse.darts (List.map SimpleGraph.Dart.symm p.darts).reverse
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*, Sym2.eq_swap]
                  /-
                    🎉 no goals
                  -/


theorem mem_darts_reverse {u v : V} {d : G.Dart} {p : G.Walk u v} :
                                                 /-
                                                   V : Type u
                                                   G : SimpleGraph V
                                                   u v : V
                                                   d : G.Dart
                                                   p : G.Walk u v
                                                   ⊢ Iff (Membership.mem p.reverse.darts d) (Membership.mem p.darts d.symm)
                                                 -/
    d ∈ p.reverse.darts ↔ d.symm ∈ p.darts := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem cons_map_snd_darts {u v : V} (p : G.Walk u v) : (u :: p.darts.map (·.snd)) = p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq (List.cons u (List.map (fun x => x.toProd.2) p.darts)) p.support
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp! [*]
                  /-
                    🎉 no goals
                  -/


theorem map_snd_darts {u v : V} (p : G.Walk u v) : p.darts.map (·.snd) = p.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq (List.map (fun x => x.toProd.2) p.darts) p.support.tail
  -/
  simpa using congr_arg List.tail (cons_map_snd_darts p)
  /-
    🎉 no goals
  -/


theorem map_fst_darts_append {u v : V} (p : G.Walk u v) :
    p.darts.map (·.fst) ++ [v] = p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq (HAppend.hAppend (List.map (fun x => x.toProd.1) p.darts) (List.cons v Li …
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp! [*]
                  /-
                    🎉 no goals
                  -/


theorem map_fst_darts {u v : V} (p : G.Walk u v) : p.darts.map (·.fst) = p.support.dropLast := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq (List.map (fun x => x.toProd.1) p.darts) p.support.dropLast
  -/
  simpa! using congr_arg List.dropLast (map_fst_darts_append p)
  /-
    🎉 no goals
  -/


@[simp]
theorem head_darts_fst {G : SimpleGraph V} {a b : V} (p : G.Walk a b) (hp : p.darts ≠ []) :
    (p.darts.head hp).fst = a := by
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    p : G.Walk a b
    hp : Ne p.darts List.nil
    ⊢ Eq (p.darts.head hp).toProd.1 a
  -/
  cases p
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      a : V
      hp : Ne SimpleGraph.Walk.nil.darts List.nil
      ⊢ Eq (SimpleGraph.Walk.nil.darts.head hp).toProd.1 a
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      a b v✝ : V
      h✝ : G.Adj a v✝
      p✝ : G.Walk v✝ b
      hp : Ne (SimpleGraph.Walk.cons h✝ p✝).darts List.nil
      ⊢ Eq ((SimpleGraph.Walk.cons h✝ p✝).darts.head hp).toProd.1 a
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem getLast_darts_snd {G : SimpleGraph V} {a b : V} (p : G.Walk a b) (hp : p.darts ≠ []) :
    (p.darts.getLast hp).snd = b := by
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    p : G.Walk a b
    hp : Ne p.darts List.nil
    ⊢ Eq (p.darts.getLast hp).toProd.2 b
  -/
  rw [← List.getLast_map (f := fun x : G.Dart ↦ x.snd)]
    /-
      V : Type u
      G : SimpleGraph V
      a b : V
      p : G.Walk a b
      hp : Ne p.darts List.nil
      ⊢ Eq ((List.map (fun x => x.toProd.2) p.darts).getLast ?h) b
    -/
  · simp_rw [p.map_snd_darts, List.getLast_tail, p.getLast_support]
    /-
      🎉 no goals
    -/
    /-
      case h
      V : Type u
      G : SimpleGraph V
      a b : V
      p : G.Walk a b
      hp : Ne p.darts List.nil
      ⊢ Ne (List.map (fun x => x.toProd.2) p.darts) List.nil
    -/
  · simpa
    /-
      🎉 no goals
    -/


@[simp]
theorem edges_nil {u : V} : (nil : G.Walk u u).edges = [] := rfl


@[simp]
theorem edges_cons {u v w : V} (h : G.Adj u v) (p : G.Walk v w) :
    (cons h p).edges = s(u, v) :: p.edges := rfl


@[simp]
theorem edges_concat {u v w : V} (p : G.Walk u v) (h : G.Adj v w) :
                                                      /-
                                                        V : Type u
                                                        G : SimpleGraph V
                                                        u v w : V
                                                        p : G.Walk u v
                                                        h : G.Adj v w
                                                        ⊢ Eq (p.concat h).edges (p.edges.concat (Sym2.mk { fst := v, snd := w }))
                                                      -/
    (p.concat h).edges = p.edges.concat s(v, w) := by simp [edges]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem edges_copy {u v u' v'} (p : G.Walk u v) (hu : u = u') (hv : v = v') :
    (p.copy hu hv).edges = p.edges := by
  /-
    V : Type u
    G : SimpleGraph V
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (p.copy hu hv).edges p.edges
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (p.copy ⋯ ⋯).edges p.edges
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem edges_append {u v w : V} (p : G.Walk u v) (p' : G.Walk v w) :
                                                    /-
                                                      V : Type u
                                                      G : SimpleGraph V
                                                      u v w : V
                                                      p : G.Walk u v
                                                      p' : G.Walk v w
                                                      ⊢ Eq (p.append p').edges (HAppend.hAppend p.edges p'.edges)
                                                    -/
    (p.append p').edges = p.edges ++ p'.edges := by simp [edges]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem edges_reverse {u v : V} (p : G.Walk u v) : p.reverse.edges = p.edges.reverse := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq p.reverse.edges p.edges.reverse
  -/
  simp [edges, List.map_reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem length_support {u v : V} (p : G.Walk u v) : p.support.length = p.length + 1 := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq p.support.length (HAdd.hAdd p.length 1)
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem length_darts {u v : V} (p : G.Walk u v) : p.darts.length = p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq p.darts.length p.length
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
                                                                                  /-
                                                                                    V : Type u
                                                                                    G : SimpleGraph V
                                                                                    u v : V
                                                                                    p : G.Walk u v
                                                                                    ⊢ Eq p.edges.length p.length
                                                                                  -/
theorem length_edges {u v : V} (p : G.Walk u v) : p.edges.length = p.length := by simp [edges]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem dart_fst_mem_support_of_mem_darts {u v : V} :
    ∀ (p : G.Walk u v) {d : G.Dart}, d ∈ p.darts → d.fst ∈ p.support
  | cons h p', d, hd => by
    /-
      V : Type u
      G : SimpleGraph V
      u v v✝ : V
      h : G.Adj u v✝
      p' : G.Walk v✝ v
      d : G.Dart
      hd : Membership.mem (SimpleGraph.Walk.cons h p').darts d
      ⊢ Membership.mem (SimpleGraph.Walk.cons h p').support d.toProd.1
    -/
    simp only [support_cons, darts_cons, List.mem_cons] at hd ⊢
    /-
      V : Type u
      G : SimpleGraph V
      u v v✝ : V
      h : G.Adj u v✝
      p' : G.Walk v✝ v
      d : G.Dart
      hd : Or (Eq d { fst := u, snd := v✝, adj := h }) (Membership.mem p'.darts d)
      ⊢ Or (Eq d.toProd.1 u) (Membership.mem p'.support d.toProd.1)
    -/
    rcases hd with (rfl | hd)
      /-
        case inl
        V : Type u
        G : SimpleGraph V
        u v v✝ : V
        h : G.Adj u v✝
        p' : G.Walk v✝ v
        ⊢ Or (Eq { fst := u, snd := v✝, adj := h }.toProd.1 u) (Membership.mem p'.supp …
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case inr
        V : Type u
        G : SimpleGraph V
        u v v✝ : V
        h : G.Adj u v✝
        p' : G.Walk v✝ v
        d : G.Dart
        hd : Membership.mem p'.darts d
        ⊢ Or (Eq d.toProd.1 u) (Membership.mem p'.support d.toProd.1)
      -/
    · exact Or.inr (dart_fst_mem_support_of_mem_darts _ hd)
      /-
        🎉 no goals
      -/


theorem dart_snd_mem_support_of_mem_darts {u v : V} (p : G.Walk u v) {d : G.Dart}
    (h : d ∈ p.darts) : d.snd ∈ p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    d : G.Dart
    h : Membership.mem p.darts d
    ⊢ Membership.mem p.support d.toProd.2
  -/
  simpa using p.reverse.dart_fst_mem_support_of_mem_darts (by simp [h] : d.symm ∈ p.reverse.darts)
  /-
    🎉 no goals
  -/


theorem fst_mem_support_of_mem_edges {t u v w : V} (p : G.Walk v w) (he : s(t, u) ∈ p.edges) :
    t ∈ p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk v w
    he : Membership.mem p.edges (Sym2.mk { fst := t, snd := u })
    ⊢ Membership.mem p.support t
  -/
  obtain ⟨d, hd, he⟩ := List.mem_map.mp he
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk v w
    he✝ : Membership.mem p.edges (Sym2.mk { fst := t, snd := u })
    d : G.Dart
    hd : Membership.mem p.darts d
    he : Eq d.edge (Sym2.mk { fst := t, snd := u })
    ⊢ Membership.mem p.support t
  -/
  rw [dart_edge_eq_mk'_iff'] at he
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk v w
    he✝ : Membership.mem p.edges (Sym2.mk { fst := t, snd := u })
    d : G.Dart
    hd : Membership.mem p.darts d
    he : Or (And (Eq d.toProd.1 t) (Eq d.toProd.2 u)) (And (Eq d.toProd.1 u) (Eq d …
    ⊢ Membership.mem p.support t
  -/
  rcases he with (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
    /-
      case intro.intro.inl.intro
      V : Type u
      G : SimpleGraph V
      v w : V
      p : G.Walk v w
      d : G.Dart
      hd : Membership.mem p.darts d
      he : Membership.mem p.edges (Sym2.mk { fst := d.toProd.1, snd := d.toProd.2 })
      ⊢ Membership.mem p.support d.toProd.1
    -/
  · exact dart_fst_mem_support_of_mem_darts _ hd
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.intro
      V : Type u
      G : SimpleGraph V
      v w : V
      p : G.Walk v w
      d : G.Dart
      hd : Membership.mem p.darts d
      he : Membership.mem p.edges (Sym2.mk { fst := d.toProd.2, snd := d.toProd.1 })
      ⊢ Membership.mem p.support d.toProd.2
    -/
  · exact dart_snd_mem_support_of_mem_darts _ hd
    /-
      🎉 no goals
    -/


theorem snd_mem_support_of_mem_edges {t u v w : V} (p : G.Walk v w) (he : s(t, u) ∈ p.edges) :
    u ∈ p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk v w
    he : Membership.mem p.edges (Sym2.mk { fst := t, snd := u })
    ⊢ Membership.mem p.support u
  -/
  rw [Sym2.eq_swap] at he
  /-
    V : Type u
    G : SimpleGraph V
    t u v w : V
    p : G.Walk v w
    he : Membership.mem p.edges (Sym2.mk { fst := u, snd := t })
    ⊢ Membership.mem p.support u
  -/
  exact p.fst_mem_support_of_mem_edges he
  /-
    🎉 no goals
  -/


theorem darts_nodup_of_support_nodup {u v : V} {p : G.Walk u v} (h : p.support.Nodup) :
    p.darts.Nodup := by
  induction p with
  | nil => simp
  | cons _ p' ih =>
    simp only [darts_cons, support_cons, List.nodup_cons] at h ⊢
    exact ⟨fun h' => h.1 (dart_fst_mem_support_of_mem_darts p' h'), ih h.2⟩


theorem edges_nodup_of_support_nodup {u v : V} {p : G.Walk u v} (h : p.support.Nodup) :
    p.edges.Nodup := by
  induction p with
  | nil => simp
  | cons _ p' ih =>
    simp only [edges_cons, support_cons, List.nodup_cons] at h ⊢
    exact ⟨fun h' => h.1 (fst_mem_support_of_mem_edges p' h'), ih h.2⟩


theorem nodup_tail_support_reverse {u : V} {p : G.Walk u u} :
    p.reverse.support.tail.Nodup ↔ p.support.tail.Nodup := by
  /-
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    ⊢ Iff p.reverse.support.tail.Nodup p.support.tail.Nodup
  -/
  rw [Walk.support_reverse]
  /-
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    ⊢ Iff p.support.reverse.tail.Nodup p.support.tail.Nodup
  -/
  refine List.nodup_tail_reverse p.support ?h
  rw [← getVert_eq_support_get? _ (by omega), List.getLast?_eq_getElem?,
    ← getVert_eq_support_get? _ (by rw [Walk.length_support]; omega)]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    u : V
    p : G.Walk u u
    ⊢ Eq (Option.some (p.getVert 0)) (Option.some (p.getVert (HSub.hSub p.support. …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem edges_injective {u v : V} : Function.Injective (Walk.edges : G.Walk u v → List (Sym2 V))
  | .nil, .nil, _ => rfl
                             /-
                               V : Type u
                               G : SimpleGraph V
                               u✝ v u v✝ : V
                               h✝ : G.Adj u v✝
                               p✝ : G.Walk v✝ u
                               h : Eq SimpleGraph.Walk.nil.edges (SimpleGraph.Walk.cons h✝ p✝).edges
                               ⊢ Eq SimpleGraph.Walk.nil (SimpleGraph.Walk.cons h✝ p✝)
                             -/
  | .nil, .cons _ _, h => by simp at h
                             /-
                               🎉 no goals
                             -/
                             /-
                               V : Type u
                               G : SimpleGraph V
                               u✝ v u v✝ : V
                               h✝ : G.Adj u v✝
                               p✝ : G.Walk v✝ u
                               h : Eq (SimpleGraph.Walk.cons h✝ p✝).edges SimpleGraph.Walk.nil.edges
                               ⊢ Eq (SimpleGraph.Walk.cons h✝ p✝) SimpleGraph.Walk.nil
                             -/
  | .cons _ _, .nil, h => by simp at h
                             /-
                               🎉 no goals
                             -/
  | .cons' u v c h₁ w₁, .cons' _ v' _ h₂ w₂, h => by
    /-
      V : Type u
      G : SimpleGraph V
      u✝ v✝ u c v : V
      h₁ : G.Adj u v
      w₁ : G.Walk v c
      v' : V
      h₂ : G.Adj u v'
      w₂ : G.Walk v' c
      h : Eq (SimpleGraph.Walk.cons' u v c h₁ w₁).edges (SimpleGraph.Walk.cons' u v' …
      ⊢ Eq (SimpleGraph.Walk.cons' u v c h₁ w₁) (SimpleGraph.Walk.cons' u v' c h₂ w₂)
    -/
    have h₃ : u ≠ v' := by rintro rfl; exact G.loopless _ h₂
    /-
      V : Type u
      G : SimpleGraph V
      u✝ v✝ u c v : V
      h₁ : G.Adj u v
      w₁ : G.Walk v c
      v' : V
      h₂ : G.Adj u v'
      w₂ : G.Walk v' c
      h : Eq (SimpleGraph.Walk.cons' u v c h₁ w₁).edges (SimpleGraph.Walk.cons' u v' …
      h₃ : Ne u v'
      ⊢ Eq (SimpleGraph.Walk.cons' u v c h₁ w₁) (SimpleGraph.Walk.cons' u v' c h₂ w₂)
    -/
    obtain ⟨rfl, h₃⟩ : v = v' ∧ w₁.edges = w₂.edges := by simpa [h₁, h₃] using h
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      u✝ v✝ u c v : V
      h₁ : G.Adj u v
      w₁ : G.Walk v c
      h₂ : G.Adj u v
      w₂ : G.Walk v c
      h : Eq (SimpleGraph.Walk.cons' u v c h₁ w₁).edges (SimpleGraph.Walk.cons' u v  …
      h₃✝ : Ne u v
      h₃ : Eq w₁.edges w₂.edges
      ⊢ Eq (SimpleGraph.Walk.cons' u v c h₁ w₁) (SimpleGraph.Walk.cons' u v c h₂ w₂)
    -/
    obtain rfl := Walk.edges_injective h₃
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      u✝ v✝ u c v : V
      h₁ : G.Adj u v
      w₁ : G.Walk v c
      h₂ : G.Adj u v
      h₃✝ : Ne u v
      h : Eq (SimpleGraph.Walk.cons' u v c h₁ w₁).edges (SimpleGraph.Walk.cons' u v  …
      h₃ : Eq w₁.edges w₁.edges
      ⊢ Eq (SimpleGraph.Walk.cons' u v c h₁ w₁) (SimpleGraph.Walk.cons' u v c h₂ w₁)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem darts_injective {u v : V} : Function.Injective (Walk.darts : G.Walk u v → List G.Dart) :=
  edges_injective.of_comp


/-- Predicate for the empty walk.

Solves the dependent type problem where `p = G.Walk.nil` typechecks
only if `p` has defeq endpoints. -/
inductive Nil : {v w : V} → G.Walk v w → Prop
  | nil {u : V} : Nil (nil : G.Walk u u)


@[simp] lemma nil_nil : (nil : G.Walk u u).Nil := Nil.nil


@[simp] lemma not_nil_cons {h : G.Adj u v} {p : G.Walk v w} : ¬ (cons h p).Nil := nofun


instance (p : G.Walk v w) : Decidable p.Nil :=
  match p with
  | nil => isTrue .nil
  | cons _ _ => isFalse nofun


protected lemma Nil.eq {p : G.Walk v w} : p.Nil → v = w | .nil => rfl


lemma not_nil_of_ne {p : G.Walk v w} : v ≠ w → ¬ p.Nil := mt Nil.eq


lemma nil_iff_support_eq {p : G.Walk v w} : p.Nil ↔ p.support = [v] := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    p : G.Walk v w
    ⊢ Iff p.Nil (Eq p.support (List.cons v List.nil))
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp
              /-
                🎉 no goals
              -/


lemma nil_iff_length_eq {p : G.Walk v w} : p.Nil ↔ p.length = 0 := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    p : G.Walk v w
    ⊢ Iff p.Nil (Eq p.length 0)
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp
              /-
                🎉 no goals
              -/


lemma not_nil_iff_lt_length {p : G.Walk v w} : ¬ p.Nil ↔ 0 < p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    p : G.Walk v w
    ⊢ Iff (Not p.Nil) (LT.lt 0 p.length)
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp
              /-
                🎉 no goals
              -/


lemma not_nil_iff {p : G.Walk v w} :
    ¬ p.Nil ↔ ∃ (u : V) (h : G.Adj v u) (q : G.Walk u w), p = cons h q := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    p : G.Walk v w
    ⊢ Iff (Not p.Nil) (Exists fun u => Exists fun h => Exists fun q => Eq p (Simpl …
  -/
              /-
                🎉 no goals
              -/
  cases p <;> simp [*]
              /-
                🎉 no goals
              -/


/-- A walk with its endpoints defeq is `Nil` if and only if it is equal to `nil`. -/
lemma nil_iff_eq_nil : ∀ {p : G.Walk v v}, p.Nil ↔ p = nil
                           /-
                             V : Type u
                             G : SimpleGraph V
                             v : V
                             ⊢ Iff SimpleGraph.Walk.nil.Nil (Eq SimpleGraph.Walk.nil SimpleGraph.Walk.nil)
                           -/
                           /-
                             🎉 no goals
                           -/
  | .nil | .cons _ _ => by simp
                           /-
                             🎉 no goals
                           -/


alias ⟨Nil.eq_nil, _⟩ := nil_iff_eq_nil


@[elab_as_elim]
def notNilRec {motive : {u w : V} → (p : G.Walk u w) → (h : ¬ p.Nil) → Sort*}
    (cons : {u v w : V} → (h : G.Adj u v) → (q : G.Walk v w) → motive (cons h q) not_nil_cons)
    (p : G.Walk u w) : (hp : ¬ p.Nil) → motive p hp :=
  match p with
  | nil => fun hp => absurd .nil hp
  | .cons h q => fun _ => cons h q


@[simp]
lemma notNilRec_cons {motive : {u w : V} → (p : G.Walk u w) → ¬ p.Nil → Sort*}
    (cons : {u v w : V} → (h : G.Adj u v) → (q : G.Walk v w) →
    motive (q.cons h) Walk.not_nil_cons) (h' : G.Adj u v) (q' : G.Walk v w) :
                                                          /-
                                                            V : Type u
                                                            G : SimpleGraph V
                                                            u v w : V
                                                            motive : {u w : V} → (p : G.Walk u w) → Not p.Nil → Sort u_1
                                                            cons : {u v w : V} → (h : G.Adj u v) → (q : G.Walk v w) → motive (SimpleGraph. …
                                                            h' : G.Adj u v
                                                            q' : G.Walk v w
                                                            ⊢ Eq (SimpleGraph.Walk.notNilRec (fun {u v w} => cons) (SimpleGraph.Walk.cons  …
                                                          -/
    @Walk.notNilRec _ _ _ _ _ cons _ _ = cons h' q' := by rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp] lemma adj_getVert_one {p : G.Walk v w} (hp : ¬ p.Nil) :
    G.Adj v (p.getVert 1) := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    p : G.Walk v w
    hp : Not p.Nil
    ⊢ G.Adj v (p.getVert 1)
  -/
  simpa using adj_getVert_succ p (by simpa [not_nil_iff_lt_length] using hp : 0 < p.length)
  /-
    🎉 no goals
  -/


/-- The walk obtained by removing the first `n` darts of a walk. -/
def drop {u v : V} (p : G.Walk u v) (n : ℕ) : G.Walk (p.getVert n) v :=
  match p, n with
  | .nil, _ => .nil
  | p, 0 => p.copy (getVert_zero p).symm rfl
  | .cons h q, (n + 1) => (q.drop n).copy (getVert_cons_succ _ h).symm rfl


/-- The walk obtained by removing the first dart of a non-nil walk. -/
def tail (p : G.Walk u v) : G.Walk (p.getVert 1) v := p.drop 1


@[simp]
                                                                           /-
                                                                             V : Type u
                                                                             G : SimpleGraph V
                                                                             u v : V
                                                                             h : G.Adj u v
                                                                             ⊢ Eq (SimpleGraph.Walk.cons h SimpleGraph.Walk.nil).tail SimpleGraph.Walk.nil
                                                                           -/
lemma tail_cons_nil (h : G.Adj u v) : (Walk.cons h .nil).tail = .nil := by rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


lemma tail_cons_eq (h : G.Adj u v) (p : G.Walk v w) :
    (p.cons h).tail = p.copy (getVert_zero p).symm rfl := by
  match p with
  | .nil => rfl
  | .cons h q => rfl


/-- The first dart of a walk. -/
@[simps]
def firstDart (p : G.Walk v w) (hp : ¬ p.Nil) : G.Dart where
  fst := v
  snd := p.getVert 1
  adj := p.adj_getVert_one hp


lemma edge_firstDart (p : G.Walk v w) (hp : ¬ p.Nil) :
    (p.firstDart hp).edge = s(v, p.getVert 1) := rfl


lemma cons_tail_eq (p : G.Walk x y) (hp : ¬ p.Nil) :
    cons (p.adj_getVert_one hp) p.tail = p := by
  cases p with
  | nil => simp only [nil_nil, not_true_eq_false] at hp
  | cons h q =>
    simp only [getVert_cons_succ, tail_cons_eq, cons_copy, copy_rfl_rfl]


@[simp] lemma cons_support_tail (p : G.Walk x y) (hp : ¬p.Nil) :
    x :: p.tail.support = p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    x y : V
    p : G.Walk x y
    hp : Not p.Nil
    ⊢ Eq (List.cons x p.tail.support) p.support
  -/
  rw [← support_cons, cons_tail_eq _ hp]
  /-
    🎉 no goals
  -/


@[simp] lemma length_tail_add_one {p : G.Walk x y} (hp : ¬ p.Nil) :
    p.tail.length + 1 = p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    x y : V
    p : G.Walk x y
    hp : Not p.Nil
    ⊢ Eq (HAdd.hAdd p.tail.length 1) p.length
  -/
  rw [← length_cons, cons_tail_eq _ hp]
  /-
    🎉 no goals
  -/


@[simp] lemma nil_copy {x' y' : V} {p : G.Walk x y} (hx : x = x') (hy : y = y') :
    (p.copy hx hy).Nil = p.Nil := by
  /-
    V : Type u
    G : SimpleGraph V
    x y x' y' : V
    p : G.Walk x y
    hx : Eq x x'
    hy : Eq y y'
    ⊢ Eq (p.copy hx hy).Nil p.Nil
  -/
  subst_vars; rfl
              /-
                🎉 no goals
              -/


@[simp] lemma support_tail (p : G.Walk v v) (hp : ¬ p.Nil) :
    p.tail.support = p.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    p : G.Walk v v
    hp : Not p.Nil
    ⊢ Eq p.tail.support p.support.tail
  -/
  rw [← cons_support_tail p hp, List.tail_cons]
  /-
    🎉 no goals
  -/


@[simp]
lemma tail_cons {t u v} (p : G.Walk u v) (h : G.Adj t u) :
    (p.cons h).tail = p.copy (getVert_zero p).symm rfl := by
  match p with
  | .nil => rfl
  | .cons h q => rfl


lemma support_tail_of_not_nil (p : G.Walk u v) (hnp : ¬p.Nil) :
    p.tail.support = p.support.tail := by
  match p with
  | .nil => simp only [nil_nil, not_true_eq_false] at hnp
  | .cons h q =>
    simp only [tail_cons, getVert_cons_succ, support_copy, support_cons, List.tail_cons]


/-- Given a vertex in the support of a path, give the path up until (and including) that vertex. -/
def takeUntil {v w : V} : ∀ (p : G.Walk v w) (u : V), u ∈ p.support → G.Walk v u
                    /-
                      V : Type u
                      V' : Type v
                      V'' : Type w
                      G : SimpleGraph V
                      G' : SimpleGraph V'
                      G'' : SimpleGraph V''
                      u✝ v✝ w✝ x y : V
                      inst✝ : DecidableEq V
                      v w u : V
                      h : Membership.mem SimpleGraph.Walk.nil.support u
                      ⊢ G.Walk v u
                    -/
  | nil, u, h => by rw [mem_support_nil_iff.mp h]
                    /-
                      🎉 no goals
                    -/
  | cons r p, u, h =>
    if hx : v = u then
         /-
           V : Type u
           V' : Type v
           V'' : Type w
           G : SimpleGraph V
           G' : SimpleGraph V'
           G'' : SimpleGraph V''
           u✝ v✝¹ w✝¹ x y : V
           inst✝ : DecidableEq V
           v w✝ w v✝ : V
           r : G.Adj v v✝
           p : G.Walk v✝ w
           u : V
           h : Membership.mem (SimpleGraph.Walk.cons r p).support u
           hx : Eq v u
           ⊢ G.Walk v u
         -/
      by subst u; exact Walk.nil
                  /-
                    🎉 no goals
                  -/
    else
      cons r (takeUntil p u <| by
        /-
          V : Type u
          V' : Type v
          V'' : Type w
          G : SimpleGraph V
          G' : SimpleGraph V'
          G'' : SimpleGraph V''
          u✝ v✝¹ w✝¹ x y : V
          inst✝ : DecidableEq V
          v w✝ w v✝ : V
          r : G.Adj v v✝
          p : G.Walk v✝ w
          u : V
          h : Membership.mem (SimpleGraph.Walk.cons r p).support u
          hx : Not (Eq v u)
          ⊢ Membership.mem p.support u
        -/
        cases h
          /-
            case head
            V : Type u
            V' : Type v
            V'' : Type w
            G : SimpleGraph V
            G' : SimpleGraph V'
            G'' : SimpleGraph V''
            u v✝¹ w✝¹ x y : V
            inst✝ : DecidableEq V
            v w✝ w v✝ : V
            r : G.Adj v v✝
            p : G.Walk v✝ w
            hx : Not (Eq v v)
            ⊢ Membership.mem p.support v
          -/
        · exact (hx rfl).elim
          /-
            🎉 no goals
          -/
          /-
            case tail
            V : Type u
            V' : Type v
            V'' : Type w
            G : SimpleGraph V
            G' : SimpleGraph V'
            G'' : SimpleGraph V''
            u✝ v✝¹ w✝¹ x y : V
            inst✝ : DecidableEq V
            v w✝ w v✝ : V
            r : G.Adj v v✝
            p : G.Walk v✝ w
            u : V
            hx : Not (Eq v u)
            a✝ : List.Mem u p.support
            ⊢ Membership.mem p.support u
          -/
        · assumption)
          /-
            🎉 no goals
          -/


/-- Given a vertex in the support of a path, give the path from (and including) that vertex to
the end. In other words, drop vertices from the front of a path until (and not including)
that vertex. -/
def dropUntil {v w : V} : ∀ (p : G.Walk v w) (u : V), u ∈ p.support → G.Walk u w
                    /-
                      V : Type u
                      V' : Type v
                      V'' : Type w
                      G : SimpleGraph V
                      G' : SimpleGraph V'
                      G'' : SimpleGraph V''
                      u✝ v✝ w✝ x y : V
                      inst✝ : DecidableEq V
                      v w u : V
                      h : Membership.mem SimpleGraph.Walk.nil.support u
                      ⊢ G.Walk u v
                    -/
  | nil, u, h => by rw [mem_support_nil_iff.mp h]
                    /-
                      🎉 no goals
                    -/
  | cons r p, u, h =>
    if hx : v = u then by
      /-
        V : Type u
        V' : Type v
        V'' : Type w
        G : SimpleGraph V
        G' : SimpleGraph V'
        G'' : SimpleGraph V''
        u✝ v✝¹ w✝¹ x y : V
        inst✝ : DecidableEq V
        v w✝ w v✝ : V
        r : G.Adj v v✝
        p : G.Walk v✝ w
        u : V
        h : Membership.mem (SimpleGraph.Walk.cons r p).support u
        hx : Eq v u
        ⊢ G.Walk u w
      -/
      subst u
      /-
        V : Type u
        V' : Type v
        V'' : Type w
        G : SimpleGraph V
        G' : SimpleGraph V'
        G'' : SimpleGraph V''
        u v✝¹ w✝¹ x y : V
        inst✝ : DecidableEq V
        v w✝ w v✝ : V
        r : G.Adj v v✝
        p : G.Walk v✝ w
        h : Membership.mem (SimpleGraph.Walk.cons r p).support v
        ⊢ G.Walk v w
      -/
      exact cons r p
      /-
        🎉 no goals
      -/
    else dropUntil p u <| by
      /-
        V : Type u
        V' : Type v
        V'' : Type w
        G : SimpleGraph V
        G' : SimpleGraph V'
        G'' : SimpleGraph V''
        u✝ v✝¹ w✝¹ x y : V
        inst✝ : DecidableEq V
        v w✝ w v✝ : V
        r : G.Adj v v✝
        p : G.Walk v✝ w
        u : V
        h : Membership.mem (SimpleGraph.Walk.cons r p).support u
        hx : Not (Eq v u)
        ⊢ Membership.mem p.support u
      -/
      cases h
        /-
          case head
          V : Type u
          V' : Type v
          V'' : Type w
          G : SimpleGraph V
          G' : SimpleGraph V'
          G'' : SimpleGraph V''
          u v✝¹ w✝¹ x y : V
          inst✝ : DecidableEq V
          v w✝ w v✝ : V
          r : G.Adj v v✝
          p : G.Walk v✝ w
          hx : Not (Eq v v)
          ⊢ Membership.mem p.support v
        -/
      · exact (hx rfl).elim
        /-
          🎉 no goals
        -/
        /-
          case tail
          V : Type u
          V' : Type v
          V'' : Type w
          G : SimpleGraph V
          G' : SimpleGraph V'
          G'' : SimpleGraph V''
          u✝ v✝¹ w✝¹ x y : V
          inst✝ : DecidableEq V
          v w✝ w v✝ : V
          r : G.Adj v v✝
          p : G.Walk v✝ w
          u : V
          hx : Not (Eq v u)
          a✝ : List.Mem u p.support
          ⊢ Membership.mem p.support u
        -/
      · assumption
        /-
          🎉 no goals
        -/


/-- The `takeUntil` and `dropUntil` functions split a walk into two pieces.
The lemma `SimpleGraph.Walk.count_support_takeUntil_eq_one` specifies where this split occurs. -/
@[simp]
theorem take_spec {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.takeUntil u h).append (p.dropUntil u h) = p := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    ⊢ Eq ((p.takeUntil u h).append (p.dropUntil u h)) p
  -/
  induction p
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w u✝ : V
      h : Membership.mem SimpleGraph.Walk.nil.support u
      ⊢ Eq ((SimpleGraph.Walk.nil.takeUntil u h).append (SimpleGraph.Walk.nil.dropUn …
    -/
  · rw [mem_support_nil_iff] at h
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w u✝ : V
      h✝ : Membership.mem SimpleGraph.Walk.nil.support u
      h : Eq u u✝
      ⊢ Eq ((SimpleGraph.Walk.nil.takeUntil u h✝).append (SimpleGraph.Walk.nil.dropU …
    -/
    subst u
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      v w u✝ : V
      h : Membership.mem SimpleGraph.Walk.nil.support u✝
      ⊢ Eq ((SimpleGraph.Walk.nil.takeUntil u✝ h).append (SimpleGraph.Walk.nil.dropU …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w u✝ v✝ w✝ : V
      h✝ : G.Adj u✝ v✝
      p✝ : G.Walk v✝ w✝
      p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq ((p✝.takeUntil u h).append (p✝ …
      h : Membership.mem (SimpleGraph.Walk.cons h✝ p✝).support u
      ⊢ Eq (((SimpleGraph.Walk.cons h✝ p✝).takeUntil u h).append ((SimpleGraph.Walk. …
    -/
  · cases h
      /-
        case cons.head
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w v✝ w✝ : V
        p✝ : G.Walk v✝ w✝
        p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq ((p✝.takeUntil u h).append (p✝ …
        h✝ : G.Adj u v✝
        ⊢ Eq (((SimpleGraph.Walk.cons h✝ p✝).takeUntil u ⋯).append ((SimpleGraph.Walk. …
      -/
    · simp!
      /-
        🎉 no goals
      -/
      /-
        case cons.tail
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w u✝ v✝ w✝ : V
        h✝ : G.Adj u✝ v✝
        p✝ : G.Walk v✝ w✝
        p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq ((p✝.takeUntil u h).append (p✝ …
        a✝ : List.Mem u p✝.support
        ⊢ Eq (((SimpleGraph.Walk.cons h✝ p✝).takeUntil u ⋯).append ((SimpleGraph.Walk. …
      -/
    · simp! only
      /-
        case cons.tail
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w u✝ v✝ w✝ : V
        h✝ : G.Adj u✝ v✝
        p✝ : G.Walk v✝ w✝
        p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq ((p✝.takeUntil u h).append (p✝ …
        a✝ : List.Mem u p✝.support
        ⊢ Eq ((dite (Eq u✝ u) (fun h => Eq.rec (motive := fun x x_1 => Membership.mem  …
      -/
                                           /-
                                             🎉 no goals
                                           -/
      split_ifs with h' <;> subst_vars <;> simp [*]
                                           /-
                                             🎉 no goals
                                           -/


theorem mem_support_iff_exists_append {V : Type u} {G : SimpleGraph V} {u v w : V}
    {p : G.Walk u v} : w ∈ p.support ↔ ∃ (q : G.Walk u w) (r : G.Walk w v), p = q.append r := by
  classical
  constructor
  · exact fun h => ⟨_, _, (p.take_spec h).symm⟩
  · rintro ⟨q, r, rfl⟩
    simp only [mem_support_append_iff, end_mem_support, start_mem_support, or_self_iff]


@[simp]
theorem count_support_takeUntil_eq_one {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.takeUntil u h).support.count u = 1 := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    ⊢ Eq (List.count u (p.takeUntil u h).support) 1
  -/
  induction p
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w u✝ : V
      h : Membership.mem SimpleGraph.Walk.nil.support u
      ⊢ Eq (List.count u (SimpleGraph.Walk.nil.takeUntil u h).support) 1
    -/
  · rw [mem_support_nil_iff] at h
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w u✝ : V
      h✝ : Membership.mem SimpleGraph.Walk.nil.support u
      h : Eq u u✝
      ⊢ Eq (List.count u (SimpleGraph.Walk.nil.takeUntil u h✝).support) 1
    -/
    subst u
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      v w u✝ : V
      h : Membership.mem SimpleGraph.Walk.nil.support u✝
      ⊢ Eq (List.count u✝ (SimpleGraph.Walk.nil.takeUntil u✝ h).support) 1
    -/
    simp!
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w u✝ v✝ w✝ : V
      h✝ : G.Adj u✝ v✝
      p✝ : G.Walk v✝ w✝
      p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq (List.count u (p✝.takeUntil u  …
      h : Membership.mem (SimpleGraph.Walk.cons h✝ p✝).support u
      ⊢ Eq (List.count u ((SimpleGraph.Walk.cons h✝ p✝).takeUntil u h).support) 1
    -/
  · cases h
      /-
        case cons.head
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w v✝ w✝ : V
        p✝ : G.Walk v✝ w✝
        p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq (List.count u (p✝.takeUntil u  …
        h✝ : G.Adj u v✝
        ⊢ Eq (List.count u ((SimpleGraph.Walk.cons h✝ p✝).takeUntil u ⋯).support) 1
      -/
    · simp!
      /-
        🎉 no goals
      -/
      /-
        case cons.tail
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w u✝ v✝ w✝ : V
        h✝ : G.Adj u✝ v✝
        p✝ : G.Walk v✝ w✝
        p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq (List.count u (p✝.takeUntil u  …
        a✝ : List.Mem u p✝.support
        ⊢ Eq (List.count u ((SimpleGraph.Walk.cons h✝ p✝).takeUntil u ⋯).support) 1
      -/
    · simp! only
      /-
        case cons.tail
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w u✝ v✝ w✝ : V
        h✝ : G.Adj u✝ v✝
        p✝ : G.Walk v✝ w✝
        p_ih✝ : ∀ (h : Membership.mem p✝.support u), Eq (List.count u (p✝.takeUntil u  …
        a✝ : List.Mem u p✝.support
        ⊢ Eq (List.count u (dite (Eq u✝ u) (fun h => Eq.rec (motive := fun x x_1 => Me …
      -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
      split_ifs with h' <;> rw [eq_comm] at h' <;> subst_vars <;> simp! [*, List.count_cons]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem count_edges_takeUntil_le_one {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) (x : V) :
    (p.takeUntil u h).edges.count s(u, x) ≤ 1 := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : V
    ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) (p.takeUntil u h).edges) 1
  -/
  induction' p with u' u' v' w' ha p' ih
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w x u' : V
      h : Membership.mem SimpleGraph.Walk.nil.support u
      ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) (SimpleGraph.Walk.nil.tak …
    -/
  · rw [mem_support_nil_iff] at h
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w x u' : V
      h✝ : Membership.mem SimpleGraph.Walk.nil.support u
      h : Eq u u'
      ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) (SimpleGraph.Walk.nil.tak …
    -/
    subst u
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      v w x u' : V
      h : Membership.mem SimpleGraph.Walk.nil.support u'
      ⊢ LE.le (List.count (Sym2.mk { fst := u', snd := x }) (SimpleGraph.Walk.nil.ta …
    -/
    simp!
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      inst✝ : DecidableEq V
      u v w x u' v' w' : V
      ha : G.Adj u' v'
      p' : G.Walk v' w'
      ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
      h : Membership.mem (SimpleGraph.Walk.cons ha p').support u
      ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) ((SimpleGraph.Walk.cons h …
    -/
  · cases h
      /-
        case cons.head
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w x v' w' : V
        p' : G.Walk v' w'
        ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
        ha : G.Adj u v'
        ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) ((SimpleGraph.Walk.cons h …
      -/
    · simp!
      /-
        🎉 no goals
      -/
      /-
        case cons.tail
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w x u' v' w' : V
        ha : G.Adj u' v'
        p' : G.Walk v' w'
        ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
        a✝ : List.Mem u p'.support
        ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) ((SimpleGraph.Walk.cons h …
      -/
    · simp! only
      /-
        case cons.tail
        V : Type u
        G : SimpleGraph V
        inst✝ : DecidableEq V
        u v w x u' v' w' : V
        ha : G.Adj u' v'
        p' : G.Walk v' w'
        ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
        a✝ : List.Mem u p'.support
        ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) (dite (Eq u' u) (fun h => …
      -/
      split_ifs with h'
        /-
          case pos
          V : Type u
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v w x u' v' w' : V
          ha : G.Adj u' v'
          p' : G.Walk v' w'
          ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
          a✝ : List.Mem u p'.support
          h' : Eq u' u
          ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) (Eq.rec (motive := fun x  …
        -/
      · subst h'
        /-
          case pos
          V : Type u
          G : SimpleGraph V
          inst✝ : DecidableEq V
          v w x u' v' w' : V
          ha : G.Adj u' v'
          p' : G.Walk v' w'
          ih : ∀ (h : Membership.mem p'.support u'), LE.le (List.count (Sym2.mk { fst := …
          a✝ : List.Mem u' p'.support
          ⊢ LE.le (List.count (Sym2.mk { fst := u', snd := x }) (Eq.rec (motive := fun x …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case neg
          V : Type u
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v w x u' v' w' : V
          ha : G.Adj u' v'
          p' : G.Walk v' w'
          ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
          a✝ : List.Mem u p'.support
          h' : Not (Eq u' u)
          ⊢ LE.le (List.count (Sym2.mk { fst := u, snd := x }) (SimpleGraph.Walk.cons ha …
        -/
      · rw [edges_cons, List.count_cons]
        /-
          case neg
          V : Type u
          G : SimpleGraph V
          inst✝ : DecidableEq V
          u v w x u' v' w' : V
          ha : G.Adj u' v'
          p' : G.Walk v' w'
          ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
          a✝ : List.Mem u p'.support
          h' : Not (Eq u' u)
          ⊢ LE.le (HAdd.hAdd (List.count (Sym2.mk { fst := u, snd := x }) (p'.takeUntil  …
        -/
        split_ifs with h''
          /-
            case pos
            V : Type u
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u v w x u' v' w' : V
            ha : G.Adj u' v'
            p' : G.Walk v' w'
            ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
            a✝ : List.Mem u p'.support
            h' : Not (Eq u' u)
            h'' : Eq (BEq.beq (Sym2.mk { fst := u', snd := v' }) (Sym2.mk { fst := u, snd  …
            ⊢ LE.le (HAdd.hAdd (List.count (Sym2.mk { fst := u, snd := x }) (p'.takeUntil  …
          -/
        · simp only [beq_iff_eq, Sym2.eq, Sym2.rel_iff'] at h''
          /-
            case pos
            V : Type u
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u v w x u' v' w' : V
            ha : G.Adj u' v'
            p' : G.Walk v' w'
            ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
            a✝ : List.Mem u p'.support
            h' : Not (Eq u' u)
            h'' : Or (Eq { fst := u', snd := v' } { fst := u, snd := x }) (Eq { fst := u', …
            ⊢ LE.le (HAdd.hAdd (List.count (Sym2.mk { fst := u, snd := x }) (p'.takeUntil  …
          -/
          obtain ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ := h''
            /-
              case pos.inl.refl
              V : Type u
              G : SimpleGraph V
              inst✝ : DecidableEq V
              u v w x w' : V
              h' : Not (Eq u u)
              p' : G.Walk x w'
              ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
              a✝ : List.Mem u p'.support
              ha : G.Adj u x
              ⊢ LE.le (HAdd.hAdd (List.count (Sym2.mk { fst := u, snd := x }) (p'.takeUntil  …
            -/
          · exact (h' rfl).elim
            /-
              🎉 no goals
            -/
            /-
              case pos.inr.refl
              V : Type u
              G : SimpleGraph V
              inst✝ : DecidableEq V
              u v w x w' : V
              h' : Not (Eq { fst := u, snd := x }.2 u)
              p' : G.Walk { fst := u, snd := x }.1 w'
              ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
              a✝ : List.Mem u p'.support
              ha : G.Adj { fst := u, snd := x }.2 { fst := u, snd := x }.1
              ⊢ LE.le (HAdd.hAdd (List.count (Sym2.mk { fst := u, snd := x }) (p'.takeUntil  …
            -/
                         /-
                           🎉 no goals
                         -/
          · cases p' <;> simp!
                         /-
                           🎉 no goals
                         -/
          /-
            case neg
            V : Type u
            G : SimpleGraph V
            inst✝ : DecidableEq V
            u v w x u' v' w' : V
            ha : G.Adj u' v'
            p' : G.Walk v' w'
            ih : ∀ (h : Membership.mem p'.support u), LE.le (List.count (Sym2.mk { fst :=  …
            a✝ : List.Mem u p'.support
            h' : Not (Eq u' u)
            h'' : Not (Eq (BEq.beq (Sym2.mk { fst := u', snd := v' }) (Sym2.mk { fst := u, …
            ⊢ LE.le (HAdd.hAdd (List.count (Sym2.mk { fst := u, snd := x }) (p'.takeUntil  …
          -/
        · apply ih
          /-
            🎉 no goals
          -/


@[simp]
theorem takeUntil_copy {u v w v' w'} (p : G.Walk v w) (hv : v = v') (hw : w = w')
    (h : u ∈ (p.copy hv hw).support) :
                                                      /-
                                                        V : Type u
                                                        V' : Type v
                                                        V'' : Type w
                                                        G : SimpleGraph V
                                                        G' : SimpleGraph V'
                                                        G'' : SimpleGraph V''
                                                        u✝ v✝ w✝ x y : V
                                                        inst✝ : DecidableEq V
                                                        u v w v' w' : V
                                                        p : G.Walk v w
                                                        hv : Eq v v'
                                                        hw : Eq w w'
                                                        h : Membership.mem (p.copy hv hw).support u
                                                        ⊢ Membership.mem p.support u
                                                      -/
    (p.copy hv hw).takeUntil u h = (p.takeUntil u (by subst_vars; exact h)).copy hv rfl := by
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w v' w' : V
    p : G.Walk v w
    hv : Eq v v'
    hw : Eq w w'
    h : Membership.mem (p.copy hv hw).support u
    ⊢ Eq ((p.copy hv hw).takeUntil u h) ((p.takeUntil u ⋯).copy hv ⋯)
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v' w' : V
    p : G.Walk v' w'
    h : Membership.mem (p.copy ⋯ ⋯).support u
    ⊢ Eq ((p.copy ⋯ ⋯).takeUntil u h) ((p.takeUntil u ⋯).copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem dropUntil_copy {u v w v' w'} (p : G.Walk v w) (hv : v = v') (hw : w = w')
    (h : u ∈ (p.copy hv hw).support) :
                                                      /-
                                                        V : Type u
                                                        V' : Type v
                                                        V'' : Type w
                                                        G : SimpleGraph V
                                                        G' : SimpleGraph V'
                                                        G'' : SimpleGraph V''
                                                        u✝ v✝ w✝ x y : V
                                                        inst✝ : DecidableEq V
                                                        u v w v' w' : V
                                                        p : G.Walk v w
                                                        hv : Eq v v'
                                                        hw : Eq w w'
                                                        h : Membership.mem (p.copy hv hw).support u
                                                        ⊢ Membership.mem p.support u
                                                      -/
    (p.copy hv hw).dropUntil u h = (p.dropUntil u (by subst_vars; exact h)).copy rfl hw := by
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w v' w' : V
    p : G.Walk v w
    hv : Eq v v'
    hw : Eq w w'
    h : Membership.mem (p.copy hv hw).support u
    ⊢ Eq ((p.copy hv hw).dropUntil u h) ((p.dropUntil u ⋯).copy ⋯ hw)
  -/
  subst_vars
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v' w' : V
    p : G.Walk v' w'
    h : Membership.mem (p.copy ⋯ ⋯).support u
    ⊢ Eq ((p.copy ⋯ ⋯).dropUntil u h) ((p.dropUntil u ⋯).copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem support_takeUntil_subset {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.takeUntil u h).support ⊆ p.support := fun x hx => by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : V
    hx : Membership.mem (p.takeUntil u h).support x
    ⊢ Membership.mem p.support x
  -/
  rw [← take_spec p h, mem_support_append_iff]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : V
    hx : Membership.mem (p.takeUntil u h).support x
    ⊢ Or (Membership.mem (p.takeUntil u h).support x) (Membership.mem (p.dropUntil …
  -/
  exact Or.inl hx
  /-
    🎉 no goals
  -/


theorem support_dropUntil_subset {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.dropUntil u h).support ⊆ p.support := fun x hx => by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : V
    hx : Membership.mem (p.dropUntil u h).support x
    ⊢ Membership.mem p.support x
  -/
  rw [← take_spec p h, mem_support_append_iff]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : V
    hx : Membership.mem (p.dropUntil u h).support x
    ⊢ Or (Membership.mem (p.takeUntil u h).support x) (Membership.mem (p.dropUntil …
  -/
  exact Or.inr hx
  /-
    🎉 no goals
  -/


theorem darts_takeUntil_subset {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.takeUntil u h).darts ⊆ p.darts := fun x hx => by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : G.Dart
    hx : Membership.mem (p.takeUntil u h).darts x
    ⊢ Membership.mem p.darts x
  -/
  rw [← take_spec p h, darts_append, List.mem_append]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : G.Dart
    hx : Membership.mem (p.takeUntil u h).darts x
    ⊢ Or (Membership.mem (p.takeUntil u h).darts x) (Membership.mem (p.dropUntil u …
  -/
  exact Or.inl hx
  /-
    🎉 no goals
  -/


theorem darts_dropUntil_subset {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.dropUntil u h).darts ⊆ p.darts := fun x hx => by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : G.Dart
    hx : Membership.mem (p.dropUntil u h).darts x
    ⊢ Membership.mem p.darts x
  -/
  rw [← take_spec p h, darts_append, List.mem_append]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    x : G.Dart
    hx : Membership.mem (p.dropUntil u h).darts x
    ⊢ Or (Membership.mem (p.takeUntil u h).darts x) (Membership.mem (p.dropUntil u …
  -/
  exact Or.inr hx
  /-
    🎉 no goals
  -/


theorem edges_takeUntil_subset {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.takeUntil u h).edges ⊆ p.edges :=
  List.map_subset _ (p.darts_takeUntil_subset h)


theorem edges_dropUntil_subset {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.dropUntil u h).edges ⊆ p.edges :=
  List.map_subset _ (p.darts_dropUntil_subset h)


theorem length_takeUntil_le {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.takeUntil u h).length ≤ p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    ⊢ LE.le (p.takeUntil u h).length p.length
  -/
  have := congr_arg Walk.length (p.take_spec h)
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    this : Eq ((p.takeUntil u h).append (p.dropUntil u h)).length p.length
    ⊢ LE.le (p.takeUntil u h).length p.length
  -/
  rw [length_append] at this
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    this : Eq (HAdd.hAdd (p.takeUntil u h).length (p.dropUntil u h).length) p.length
    ⊢ LE.le (p.takeUntil u h).length p.length
  -/
  exact Nat.le.intro this
  /-
    🎉 no goals
  -/


theorem length_dropUntil_le {u v w : V} (p : G.Walk v w) (h : u ∈ p.support) :
    (p.dropUntil u h).length ≤ p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    ⊢ LE.le (p.dropUntil u h).length p.length
  -/
  have := congr_arg Walk.length (p.take_spec h)
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    this : Eq ((p.takeUntil u h).append (p.dropUntil u h)).length p.length
    ⊢ LE.le (p.dropUntil u h).length p.length
  -/
  rw [length_append, add_comm] at this
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v w : V
    p : G.Walk v w
    h : Membership.mem p.support u
    this : Eq (HAdd.hAdd (p.dropUntil u h).length (p.takeUntil u h).length) p.length
    ⊢ LE.le (p.dropUntil u h).length p.length
  -/
  exact Nat.le.intro this
  /-
    🎉 no goals
  -/


/-- Rotate a loop walk such that it is centered at the given vertex. -/
def rotate {u v : V} (c : G.Walk v v) (h : u ∈ c.support) : G.Walk u u :=
  (c.dropUntil u h).append (c.takeUntil u h)


@[simp]
theorem support_rotate {u v : V} (c : G.Walk v v) (h : u ∈ c.support) :
    (c.rotate h).support.tail ~r c.support.tail := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ (c.rotate h).support.tail.IsRotated c.support.tail
  -/
  simp only [rotate, tail_support_append]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ (HAppend.hAppend (c.dropUntil u h).support.tail (c.takeUntil u h).support.ta …
  -/
  apply List.IsRotated.trans List.isRotated_append
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ (HAppend.hAppend (c.takeUntil u h).support.tail (c.dropUntil u h).support.ta …
  -/
  rw [← tail_support_append, take_spec]
  /-
    🎉 no goals
  -/


theorem rotate_darts {u v : V} (c : G.Walk v v) (h : u ∈ c.support) :
    (c.rotate h).darts ~r c.darts := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ (c.rotate h).darts.IsRotated c.darts
  -/
  simp only [rotate, darts_append]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ (HAppend.hAppend (c.dropUntil u h).darts (c.takeUntil u h).darts).IsRotated  …
  -/
  apply List.IsRotated.trans List.isRotated_append
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    u v : V
    c : G.Walk v v
    h : Membership.mem c.support u
    ⊢ (HAppend.hAppend (c.takeUntil u h).darts (c.dropUntil u h).darts).IsRotated  …
  -/
  rw [← darts_append, take_spec]
  /-
    🎉 no goals
  -/


theorem rotate_edges {u v : V} (c : G.Walk v v) (h : u ∈ c.support) :
    (c.rotate h).edges ~r c.edges :=
  (rotate_darts c h).map _


/-- Given a set `S` and a walk `w` from `u` to `v` such that `u ∈ S` but `v ∉ S`,
there exists a dart in the walk whose start is in `S` but whose end is not. -/
theorem exists_boundary_dart {u v : V} (p : G.Walk u v) (S : Set V) (uS : u ∈ S) (vS : v ∉ S) :
    ∃ d : G.Dart, d ∈ p.darts ∧ d.fst ∈ S ∧ d.snd ∉ S := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    S : Set V
    uS : Membership.mem S u
    vS : Not (Membership.mem S v)
    ⊢ Exists fun d => And (Membership.mem p.darts d) (And (Membership.mem S d.toPr …
  -/
  induction' p with _ x y w a p' ih
    /-
      case nil
      V : Type u
      G : SimpleGraph V
      u v : V
      S : Set V
      u✝ : V
      uS : Membership.mem S u✝
      vS : Not (Membership.mem S u✝)
      ⊢ Exists fun d => And (Membership.mem SimpleGraph.Walk.nil.darts d) (And (Memb …
    -/
  · cases vS uS
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      G : SimpleGraph V
      u v : V
      S : Set V
      x y w : V
      a : G.Adj x y
      p' : G.Walk y w
      ih : Membership.mem S y → Not (Membership.mem S w) → Exists fun d => And (Memb …
      uS : Membership.mem S x
      vS : Not (Membership.mem S w)
      ⊢ Exists fun d => And (Membership.mem (SimpleGraph.Walk.cons a p').darts d) (A …
    -/
  · by_cases h : y ∈ S
      /-
        case pos
        V : Type u
        G : SimpleGraph V
        u v : V
        S : Set V
        x y w : V
        a : G.Adj x y
        p' : G.Walk y w
        ih : Membership.mem S y → Not (Membership.mem S w) → Exists fun d => And (Memb …
        uS : Membership.mem S x
        vS : Not (Membership.mem S w)
        h : Membership.mem S y
        ⊢ Exists fun d => And (Membership.mem (SimpleGraph.Walk.cons a p').darts d) (A …
      -/
    · obtain ⟨d, hd, hcd⟩ := ih h vS
      /-
        case pos.intro.intro
        V : Type u
        G : SimpleGraph V
        u v : V
        S : Set V
        x y w : V
        a : G.Adj x y
        p' : G.Walk y w
        ih : Membership.mem S y → Not (Membership.mem S w) → Exists fun d => And (Memb …
        uS : Membership.mem S x
        vS : Not (Membership.mem S w)
        h : Membership.mem S y
        d : G.Dart
        hd : Membership.mem p'.darts d
        hcd : And (Membership.mem S d.toProd.1) (Not (Membership.mem S d.toProd.2))
        ⊢ Exists fun d => And (Membership.mem (SimpleGraph.Walk.cons a p').darts d) (A …
      -/
      exact ⟨d, List.Mem.tail _ hd, hcd⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u
        G : SimpleGraph V
        u v : V
        S : Set V
        x y w : V
        a : G.Adj x y
        p' : G.Walk y w
        ih : Membership.mem S y → Not (Membership.mem S w) → Exists fun d => And (Memb …
        uS : Membership.mem S x
        vS : Not (Membership.mem S w)
        h : Not (Membership.mem S y)
        ⊢ Exists fun d => And (Membership.mem (SimpleGraph.Walk.cons a p').darts d) (A …
      -/
    · exact ⟨⟨(x, y), a⟩, List.Mem.head _, uS, h⟩
      /-
        🎉 no goals
      -/


@[simp] lemma getVert_copy  {u v w x : V} (p : G.Walk u v) (i : ℕ) (h : u = w) (h' : v = x) :
    (p.copy h h').getVert i = p.getVert i := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w x : V
    p : G.Walk u v
    i : Nat
    h : Eq u w
    h' : Eq v x
    ⊢ Eq ((p.copy h h').getVert i) (p.getVert i)
  -/
  subst_vars
  match p, i with
  | .nil, _ =>
    rw [getVert_of_length_le _ (by simp only [length_nil, Nat.zero_le] : nil.length ≤ _)]
    rw [getVert_of_length_le _ (by simp only [length_copy, length_nil, Nat.zero_le])]
  | .cons hadj q, 0 => simp only [copy_rfl_rfl, getVert_zero]
  | .cons hadj q, (n + 1) => simp only [copy_cons, getVert_cons_succ]; rfl


@[simp] lemma getVert_tail {u v n} (p : G.Walk u v) (hnp: ¬ p.Nil) :
    p.tail.getVert n = p.getVert (n + 1) := by
  match p with
  | .nil => rfl
  | .cons h q =>
    simp only [getVert_cons_succ, tail_cons_eq, getVert_cons]
    exact getVert_copy q n (getVert_zero q).symm rfl


/-- Given a walk `w` and a node in the support, there exists a natural `n`, such that given node
is the `n`-th node (zero-indexed) in the walk. In addition, `n` is at most the length of the path.
Due to the definition of `getVert` it would otherwise be legal to return a larger `n` for the last
node. -/
theorem mem_support_iff_exists_getVert {u v w : V} {p : G.Walk v w} :
    u ∈ p.support ↔ ∃ n, p.getVert n = u ∧ n ≤ p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    u v w : V
    p : G.Walk v w
    ⊢ Iff (Membership.mem p.support u) (Exists fun n => And (Eq (p.getVert n) u) ( …
  -/
  constructor
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      ⊢ Membership.mem p.support u → Exists fun n => And (Eq (p.getVert n) u) (LE.le …
    -/
  · intro h
    /-
      case mp
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      h : Membership.mem p.support u
      ⊢ Exists fun n => And (Eq (p.getVert n) u) (LE.le n p.length)
    -/
    obtain ⟨q, r, hqr⟩ := SimpleGraph.Walk.mem_support_iff_exists_append.mp h
    /-
      case mp.intro.intro
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      h : Membership.mem p.support u
      q : G.Walk v u
      r : G.Walk u w
      hqr : Eq p (q.append r)
      ⊢ Exists fun n => And (Eq (p.getVert n) u) (LE.le n p.length)
    -/
    use q.length
    /-
      case h
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      h : Membership.mem p.support u
      q : G.Walk v u
      r : G.Walk u w
      hqr : Eq p (q.append r)
      ⊢ And (Eq (p.getVert q.length) u) (LE.le q.length p.length)
    -/
    rw [hqr]
    /-
      case h
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      h : Membership.mem p.support u
      q : G.Walk v u
      r : G.Walk u w
      hqr : Eq p (q.append r)
      ⊢ And (Eq ((q.append r).getVert q.length) u) (LE.le q.length (q.append r).leng …
    -/
    rw [Walk.getVert_append]
    simp only [lt_self_iff_false, ↓reduceIte, Nat.sub_self, getVert_zero, length_append,
      Nat.le_add_right, and_self]
    /-
      case mpr
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      ⊢ (Exists fun n => And (Eq (p.getVert n) u) (LE.le n p.length)) → Membership.m …
    -/
  · rintro ⟨n, hn⟩
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      n : Nat
      hn : And (Eq (p.getVert n) u) (LE.le n p.length)
      ⊢ Membership.mem p.support u
    -/
    rw [SimpleGraph.Walk.mem_support_iff]
    /-
      case mpr.intro
      V : Type u
      G : SimpleGraph V
      u v w : V
      p : G.Walk v w
      n : Nat
      hn : And (Eq (p.getVert n) u) (LE.le n p.length)
      ⊢ Or (Eq u v) (Membership.mem p.support.tail u)
    -/
    by_cases h0 : n = 0
      /-
        case pos
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Eq n 0
        ⊢ Or (Eq u v) (Membership.mem p.support.tail u)
      -/
    · rw [h0, getVert_zero] at hn
      /-
        case pos
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq v u) (LE.le 0 p.length)
        h0 : Eq n 0
        ⊢ Or (Eq u v) (Membership.mem p.support.tail u)
      -/
      left
      /-
        case pos.h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq v u) (LE.le 0 p.length)
        h0 : Eq n 0
        ⊢ Eq u v
      -/
      exact hn.1.symm
      /-
        🎉 no goals
      -/
      /-
        case neg
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        ⊢ Or (Eq u v) (Membership.mem p.support.tail u)
      -/
    · right
      have hnp : ¬ p.Nil := by
        rw [nil_iff_length_eq]
        omega
      /-
        case neg.h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        ⊢ Membership.mem p.support.tail u
      -/
      rw [← support_tail_of_not_nil _ hnp]
      /-
        case neg.h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        ⊢ Membership.mem p.tail.support u
      -/
      rw [mem_support_iff_exists_getVert]
      /-
        case neg.h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        ⊢ Exists fun n => And (Eq (p.tail.getVert n) u) (LE.le n p.tail.length)
      -/
      use n - 1
      /-
        case h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        ⊢ And (Eq (p.tail.getVert (HSub.hSub n 1)) u) (LE.le (HSub.hSub n 1) p.tail.le …
      -/
      simp only [Nat.sub_le_iff_le_add]
      /-
        case h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        ⊢ And (Eq (p.tail.getVert (HSub.hSub n 1)) u) (LE.le n (HAdd.hAdd p.tail.lengt …
      -/
      rw [getVert_tail _ hnp, length_tail_add_one hnp]
      /-
        case h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        ⊢ And (Eq (p.getVert (HAdd.hAdd (HSub.hSub n 1) 1)) u) (LE.le n p.length)
      -/
      have : (n - 1 + 1) = n:= by omega
      /-
        case h
        V : Type u
        G : SimpleGraph V
        u v w : V
        p : G.Walk v w
        n : Nat
        hn : And (Eq (p.getVert n) u) (LE.le n p.length)
        h0 : Not (Eq n 0)
        hnp : Not p.Nil
        this : Eq (HAdd.hAdd (HSub.hSub n 1) 1) n
        ⊢ And (Eq (p.getVert (HAdd.hAdd (HSub.hSub n 1) 1)) u) (LE.le n p.length)
      -/
      rwa [this]
      /-
        🎉 no goals
      -/
termination_by p.length
decreasing_by
· simp_wf
  rw [@Nat.lt_iff_add_one_le]
  rw [length_tail_add_one hnp]


/-- Given a graph homomorphism, map walks to walks. -/
protected def map (f : G →g G') {u v : V} : G.Walk u v → G'.Walk (f u) (f v)
  | nil => nil
  | cons h p => cons (f.map_adj h) (p.map f)


@[simp]
theorem map_nil : (nil : G.Walk u u).map f = nil := rfl


@[simp]
theorem map_cons {w : V} (h : G.Adj w u) : (cons h p).map f = cons (f.map_adj h) (p.map f) := rfl


@[simp]
theorem map_copy (hu : u = u') (hv : v = v') :
    (p.copy hu hv).map f = (p.map f).copy (hu ▸ rfl) (hv ▸ rfl) := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    f : G.Hom G'
    u v u' v' : V
    p : G.Walk u v
    hu : Eq u u'
    hv : Eq v v'
    ⊢ Eq (SimpleGraph.Walk.map f (p.copy hu hv)) ((SimpleGraph.Walk.map f p).copy  …
  -/
  subst_vars
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    f : G.Hom G'
    u' v' : V
    p : G.Walk u' v'
    ⊢ Eq (SimpleGraph.Walk.map f (p.copy ⋯ ⋯)) ((SimpleGraph.Walk.map f p).copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_id (p : G.Walk u v) : p.map Hom.id = p := by
  induction p with
  | nil => rfl
  | cons _ p' ih => simp [ih]


@[simp]
theorem map_map : (p.map f).map f' = p.map (f'.comp f) := by
  induction p with
  | nil => rfl
  | cons _ _ ih => simp [ih]


/-- Unlike categories, for graphs vertex equality is an important notion, so needing to be able to
work with equality of graph homomorphisms is a necessary evil. -/
theorem map_eq_of_eq {f : G →g G'} (f' : G →g G') (h : f = f') :
    p.map f = (p.map f').copy (h ▸ rfl) (h ▸ rfl) := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    u v : V
    p : G.Walk u v
    f f' : G.Hom G'
    h : Eq f f'
    ⊢ Eq (SimpleGraph.Walk.map f p) ((SimpleGraph.Walk.map f' p).copy ⋯ ⋯)
  -/
  subst_vars
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    u v : V
    p : G.Walk u v
    f' : G.Hom G'
    ⊢ Eq (SimpleGraph.Walk.map f' p) ((SimpleGraph.Walk.map f' p).copy ⋯ ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
                                                                        /-
                                                                          V : Type u
                                                                          V' : Type v
                                                                          G : SimpleGraph V
                                                                          G' : SimpleGraph V'
                                                                          f : G.Hom G'
                                                                          u : V
                                                                          p : G.Walk u u
                                                                          ⊢ Iff (Eq (SimpleGraph.Walk.map f p) SimpleGraph.Walk.nil) (Eq p SimpleGraph.W …
                                                                        -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
theorem map_eq_nil_iff {p : G.Walk u u} : p.map f = nil ↔ p = nil := by cases p <;> simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp]
                                                       /-
                                                         V : Type u
                                                         V' : Type v
                                                         G : SimpleGraph V
                                                         G' : SimpleGraph V'
                                                         f : G.Hom G'
                                                         u v : V
                                                         p : G.Walk u v
                                                         ⊢ Eq (SimpleGraph.Walk.map f p).length p.length
                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
theorem length_map : (p.map f).length = p.length := by induction p <;> simp [*]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem map_append {u v w : V} (p : G.Walk u v) (q : G.Walk v w) :
                                                          /-
                                                            V : Type u
                                                            V' : Type v
                                                            G : SimpleGraph V
                                                            G' : SimpleGraph V'
                                                            f : G.Hom G'
                                                            u v w : V
                                                            p : G.Walk u v
                                                            q : G.Walk v w
                                                            ⊢ Eq (SimpleGraph.Walk.map f (p.append q)) ((SimpleGraph.Walk.map f p).append  …
                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    (p.append q).map f = (p.map f).append (q.map f) := by induction p <;> simp [*]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
                                                                /-
                                                                  V : Type u
                                                                  V' : Type v
                                                                  G : SimpleGraph V
                                                                  G' : SimpleGraph V'
                                                                  f : G.Hom G'
                                                                  u v : V
                                                                  p : G.Walk u v
                                                                  ⊢ Eq (SimpleGraph.Walk.map f p).reverse (SimpleGraph.Walk.map f p.reverse)
                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
theorem reverse_map : (p.map f).reverse = p.reverse.map f := by induction p <;> simp [map_append, *]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
                                                                /-
                                                                  V : Type u
                                                                  V' : Type v
                                                                  G : SimpleGraph V
                                                                  G' : SimpleGraph V'
                                                                  f : G.Hom G'
                                                                  u v : V
                                                                  p : G.Walk u v
                                                                  ⊢ Eq (SimpleGraph.Walk.map f p).support (List.map (⇑f) p.support)
                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
theorem support_map : (p.map f).support = p.support.map f := by induction p <;> simp [*]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
                                                                  /-
                                                                    V : Type u
                                                                    V' : Type v
                                                                    G : SimpleGraph V
                                                                    G' : SimpleGraph V'
                                                                    f : G.Hom G'
                                                                    u v : V
                                                                    p : G.Walk u v
                                                                    ⊢ Eq (SimpleGraph.Walk.map f p).darts (List.map f.mapDart p.darts)
                                                                  -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
theorem darts_map : (p.map f).darts = p.darts.map f.mapDart := by induction p <;> simp [*]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem edges_map : (p.map f).edges = p.edges.map (Sym2.map f) := by
  induction p with
  | nil => rfl
  | cons _ _ ih =>
    simp only [Walk.map_cons, edges_cons, List.map_cons, Sym2.map_pair_eq, List.cons.injEq,
      true_and, ih]


theorem map_injective_of_injective {f : G →g G'} (hinj : Function.Injective f) (u v : V) :
    Function.Injective (Walk.map f : G.Walk u v → G'.Walk (f u) (f v)) := by
  /-
    V : Type u
    V' : Type v
    G : SimpleGraph V
    G' : SimpleGraph V'
    f : G.Hom G'
    hinj : Function.Injective ⇑f
    u v : V
    ⊢ Function.Injective (SimpleGraph.Walk.map f)
  -/
  intro p p' h
  induction p with
  | nil =>
    cases p'
    · rfl
    · simp at h
  | cons _ _ ih =>
    cases p' with
    | nil => simp at h
    | cons _ _ =>
      simp only [map_cons, cons.injEq] at h
      cases hinj h.1
      simp only [cons.injEq, heq_iff_eq, true_and]
      apply ih
      simpa using h.2


/-- The specialization of `SimpleGraph.Walk.map` for mapping walks to supergraphs. -/
abbrev mapLe {G G' : SimpleGraph V} (h : G ≤ G') {u v : V} (p : G.Walk u v) : G'.Walk u v :=
  p.map (Hom.mapSpanningSubgraphs h)


/-- The walk `p` transferred to lie in `H`, given that `H` contains its edges. -/
@[simp]
protected def transfer {u v : V} (p : G.Walk u v)
    (H : SimpleGraph V) (h : ∀ e, e ∈ p.edges → e ∈ H.edgeSet) : H.Walk u v :=
  match p with
  | nil => nil
  | cons' u v w _ p =>
                        /-
                          V : Type u
                          V' : Type v
                          V'' : Type w
                          G : SimpleGraph V
                          G' : SimpleGraph V'
                          G'' : SimpleGraph V''
                          f : G.Hom G'
                          f' : G'.Hom G''
                          u✝¹ v✝¹ u' v' : V
                          p✝¹ : G.Walk u✝¹ v✝¹
                          u✝ v✝ : V
                          p✝ : G.Walk u✝ v✝
                          H : SimpleGraph V
                          u w v : V
                          h✝ : G.Adj u v
                          p : G.Walk v w
                          h : ∀ (e : Sym2 V), Membership.mem (SimpleGraph.Walk.cons' u v w h✝ p).edges e …
                          ⊢ Membership.mem (SimpleGraph.Walk.cons' u v w h✝ p).edges (Sym2.mk { fst := u …
                        -/
                        /-
                          🎉 no goals
                        -/
    cons (h s(u, v) (by simp)) (p.transfer H fun e he => h e (by simp [he]))
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem transfer_self : p.transfer G p.edges_subset_edgeSet = p := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    ⊢ Eq (p.transfer G ⋯) p
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


theorem transfer_eq_map_of_le (hp) (GH : G ≤ H) :
    p.transfer H hp = p.map (SimpleGraph.Hom.mapSpanningSubgraphs GH) := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    H : SimpleGraph V
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Membership.mem H.edgeSet e
    GH : LE.le G H
    ⊢ Eq (p.transfer H hp) (SimpleGraph.Walk.map (SimpleGraph.Hom.mapSpanningSubgr …
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem edges_transfer (hp) : (p.transfer H hp).edges = p.edges := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    H : SimpleGraph V
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Membership.mem H.edgeSet e
    ⊢ Eq (p.transfer H hp).edges p.edges
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem support_transfer (hp) : (p.transfer H hp).support = p.support := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    H : SimpleGraph V
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Membership.mem H.edgeSet e
    ⊢ Eq (p.transfer H hp).support p.support
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem length_transfer (hp) : (p.transfer H hp).length = p.length := by
  /-
    V : Type u
    G : SimpleGraph V
    u v : V
    p : G.Walk u v
    H : SimpleGraph V
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Membership.mem H.edgeSet e
    ⊢ Eq (p.transfer H hp).length p.length
  -/
                  /-
                    🎉 no goals
                  -/
  induction p <;> simp [*]
                  /-
                    🎉 no goals
                  -/

-- Porting note: this failed the simpNF linter since it was originally of the form
-- `(p.transfer H hp).transfer K hp' = p.transfer K hp''` with `hp'` a function of `hp` and `hp'`.
-- This was a mistake and it's corrected here.

@[simp]
theorem transfer_transfer (hp) {K : SimpleGraph V} (hp') :
    (p.transfer H hp).transfer K hp' = p.transfer K (p.edges_transfer hp ▸ hp') := by
  induction p with
  | nil => simp
  | cons _ _ ih =>
    simp only [Walk.transfer, cons.injEq, heq_eq_eq, true_and]
    apply ih


@[simp]
theorem transfer_append {w : V} (q : G.Walk v w) (hpq) :
    (p.append q).transfer H hpq =
                                          /-
                                            V : Type u
                                            V' : Type v
                                            V'' : Type w
                                            G : SimpleGraph V
                                            G' : SimpleGraph V'
                                            G'' : SimpleGraph V''
                                            f : G.Hom G'
                                            f' : G'.Hom G''
                                            u✝ v✝ u' v' : V
                                            p✝ : G.Walk u✝ v✝
                                            u v : V
                                            p : G.Walk u v
                                            H : SimpleGraph V
                                            w : V
                                            q : G.Walk v w
                                            hpq : ∀ (e : Sym2 V), Membership.mem (p.append q).edges e → Membership.mem H.e …
                                            e : Sym2 V
                                            he : Membership.mem p.edges e
                                            ⊢ Membership.mem (p.append q).edges e
                                          -/
      (p.transfer H fun e he => hpq _ (by simp [he])).append
                                          /-
                                            🎉 no goals
                                          -/
                                            /-
                                              V : Type u
                                              V' : Type v
                                              V'' : Type w
                                              G : SimpleGraph V
                                              G' : SimpleGraph V'
                                              G'' : SimpleGraph V''
                                              f : G.Hom G'
                                              f' : G'.Hom G''
                                              u✝ v✝ u' v' : V
                                              p✝ : G.Walk u✝ v✝
                                              u v : V
                                              p : G.Walk u v
                                              H : SimpleGraph V
                                              w : V
                                              q : G.Walk v w
                                              hpq : ∀ (e : Sym2 V), Membership.mem (p.append q).edges e → Membership.mem H.e …
                                              e : Sym2 V
                                              he : Membership.mem q.edges e
                                              ⊢ Membership.mem (p.append q).edges e
                                            -/
        (q.transfer H fun e he => hpq _ (by simp [he])) := by
                                            /-
                                              🎉 no goals
                                            -/
  induction p with
  | nil => simp
  | cons _ _ ih => simp only [Walk.transfer, cons_append, cons.injEq, heq_eq_eq, true_and, ih]


@[simp]
theorem reverse_transfer (hp) :
    (p.transfer H hp).reverse =
                               /-
                                 V : Type u
                                 V' : Type v
                                 V'' : Type w
                                 G : SimpleGraph V
                                 G' : SimpleGraph V'
                                 G'' : SimpleGraph V''
                                 f : G.Hom G'
                                 f' : G'.Hom G''
                                 u✝ v✝ u' v' : V
                                 p✝ : G.Walk u✝ v✝
                                 u v : V
                                 p : G.Walk u v
                                 H : SimpleGraph V
                                 hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Membership.mem H.edgeSet e
                                 ⊢ ∀ (e : Sym2 V), Membership.mem p.reverse.edges e → Membership.mem H.edgeSet e
                               -/
      p.reverse.transfer H (by simp only [edges_reverse, List.mem_reverse]; exact hp) := by
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  induction p with
  | nil => simp
  | cons _ _ ih => simp only [transfer_append, Walk.transfer, reverse_nil, reverse_cons, ih]


/-- Given a walk that avoids a set of edges, produce a walk in the graph
with those edges deleted. -/
abbrev toDeleteEdges (s : Set (Sym2 V)) {v w : V} (p : G.Walk v w)
    (hp : ∀ e, e ∈ p.edges → ¬e ∈ s) : (G.deleteEdges s).Walk v w :=
  p.transfer _ <| by
    /-
      V : Type u
      V' : Type v
      V'' : Type w
      G : SimpleGraph V
      G' : SimpleGraph V'
      G'' : SimpleGraph V''
      s : Set (Sym2 V)
      v w : V
      p : G.Walk v w
      hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
      ⊢ ∀ (e : Sym2 V), Membership.mem p.edges e → Membership.mem (G.deleteEdges s). …
    -/
    simp only [edgeSet_deleteEdges, Set.mem_diff]
    /-
      V : Type u
      V' : Type v
      V'' : Type w
      G : SimpleGraph V
      G' : SimpleGraph V'
      G'' : SimpleGraph V''
      s : Set (Sym2 V)
      v w : V
      p : G.Walk v w
      hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
      ⊢ ∀ (e : Sym2 V), Membership.mem p.edges e → And (Membership.mem G.edgeSet e)  …
    -/
    exact fun e ep => ⟨edges_subset_edgeSet p ep, hp e ep⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem toDeleteEdges_nil (s : Set (Sym2 V)) {v : V} (hp) :
    (Walk.nil : G.Walk v v).toDeleteEdges s hp = Walk.nil := rfl


@[simp]
theorem toDeleteEdges_cons (s : Set (Sym2 V)) {u v w : V} (h : G.Adj u v) (p : G.Walk v w) (hp) :
    (Walk.cons h p).toDeleteEdges s hp =
      Walk.cons (deleteEdges_adj.mpr ⟨h, hp _ (List.Mem.head _)⟩)
        (p.toDeleteEdges s fun _ he => hp _ <| List.Mem.tail _ he) :=
  rfl


/-- Given a walk that avoids an edge, create a walk in the subgraph with that edge deleted.
This is an abbreviation for `SimpleGraph.Walk.toDeleteEdges`. -/
abbrev toDeleteEdge (e : Sym2 V) (p : G.Walk v w) (hp : e ∉ p.edges) :
    (G.deleteEdges {e}).Walk v w :=
                                    /-
                                      V : Type u
                                      V' : Type v
                                      V'' : Type w
                                      G : SimpleGraph V
                                      G' : SimpleGraph V'
                                      G'' : SimpleGraph V''
                                      v w : V
                                      e : Sym2 V
                                      p : G.Walk v w
                                      hp : Not (Membership.mem p.edges e)
                                      e' : Sym2 V
                                      ⊢ Membership.mem p.edges e' → Not (Membership.mem (Singleton.singleton e) e')
                                    -/
  p.toDeleteEdges {e} (fun e' => by contrapose!; simp +contextual [hp])
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem map_toDeleteEdges_eq (s : Set (Sym2 V)) {p : G.Walk v w} (hp) :
    Walk.map (Hom.mapSpanningSubgraphs (G.deleteEdges_le s)) (p.toDeleteEdges s hp) = p := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    s : Set (Sym2 V)
    p : G.Walk v w
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
    ⊢ Eq (SimpleGraph.Walk.map (SimpleGraph.Hom.mapSpanningSubgraphs ⋯) (SimpleGra …
  -/
  rw [← transfer_eq_map_of_le, transfer_transfer, transfer_self]
  /-
    case hp'
    V : Type u
    G : SimpleGraph V
    v w : V
    s : Set (Sym2 V)
    p : G.Walk v w
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
    ⊢ ∀ (e : Sym2 V), Membership.mem (p.transfer (G.deleteEdges s) ⋯).edges e → Me …
  -/
  intros e
  /-
    case hp'
    V : Type u
    G : SimpleGraph V
    v w : V
    s : Set (Sym2 V)
    p : G.Walk v w
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
    e : Sym2 V
    ⊢ Membership.mem (p.transfer (G.deleteEdges s) ⋯).edges e → Membership.mem G.e …
  -/
  rw [edges_transfer]
  /-
    case hp'
    V : Type u
    G : SimpleGraph V
    v w : V
    s : Set (Sym2 V)
    p : G.Walk v w
    hp : ∀ (e : Sym2 V), Membership.mem p.edges e → Not (Membership.mem s e)
    e : Sym2 V
    ⊢ Membership.mem p.edges e → Membership.mem G.edgeSet e
  -/
  apply edges_subset_edgeSet p
  /-
    🎉 no goals
  -/


