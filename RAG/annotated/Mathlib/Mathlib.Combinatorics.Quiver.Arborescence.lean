/-- A quiver is an arborescence when there is a unique path from the default vertex
    to every other vertex. -/
class Arborescence (V : Type u) [Quiver.{v} V] : Type max u v where
  /-- The root of the arborescence. -/
  root : V
  /-- There is a unique path from the root to any other vertex. -/
  uniquePath : ∀ b : V, Unique (Path root b)


/-- The root of an arborescence. -/
def root (V : Type u) [Quiver V] [Arborescence V] : V :=
  Arborescence.root


instance {V : Type u} [Quiver V] [Arborescence V] (b : V) : Unique (Path (root V) b) :=
  Arborescence.uniquePath b


/-- To show that `[Quiver V]` is an arborescence with root `r : V`, it suffices to
  - provide a height function `V → ℕ` such that every arrow goes from a
    lower vertex to a higher vertex,
  - show that every vertex has at most one arrow to it, and
  - show that every vertex other than `r` has an arrow to it. -/
noncomputable def arborescenceMk {V : Type u} [Quiver V] (r : V) (height : V → ℕ)
    (height_lt : ∀ ⦃a b⦄, (a ⟶ b) → height a < height b)
    (unique_arrow : ∀ ⦃a b c : V⦄ (e : a ⟶ c) (f : b ⟶ c), a = b ∧ HEq e f)
    (root_or_arrow : ∀ b, b = r ∨ ∃ a, Nonempty (a ⟶ b)) :
    Arborescence V where
  root := r
  uniquePath b :=
    ⟨Classical.inhabited_of_nonempty (by
      /-
        V : Type u
        inst✝ : Quiver V
        r : V
        height : V → Nat
        height_lt : ∀ ⦃a b : V⦄, Quiver.Hom a b → LT.lt (height a) (height b)
        unique_arrow : ∀ ⦃a b c : V⦄ (e : Quiver.Hom a c) (f : Quiver.Hom b c), And (E …
        root_or_arrow : ∀ (b : V), Or (Eq b r) (Exists fun a => Nonempty (Quiver.Hom a …
        b : V
        ⊢ Nonempty (Quiver.Path r b)
      -/
      rcases show ∃ n, height b < n from ⟨_, Nat.lt.base _⟩ with ⟨n, hn⟩
      induction n generalizing b with
      | zero => exact False.elim (Nat.not_lt_zero _ hn)
      | succ n ih =>
      rcases root_or_arrow b with (⟨⟨⟩⟩ | ⟨a, ⟨e⟩⟩)
      · exact ⟨Path.nil⟩
      · rcases ih a (lt_of_lt_of_le (height_lt e) (Nat.lt_succ_iff.mp hn)) with ⟨p⟩
        exact ⟨p.cons e⟩), by
      have height_le : ∀ {a b}, Path a b → height a ≤ height b := by
        intro a b p
        induction p with
        | nil => rfl
        | cons _ e ih => exact le_of_lt (lt_of_le_of_lt ih (height_lt e))
      suffices ∀ p q : Path r b, p = q by
        intro p
        apply this
      /-
        V : Type u
        inst✝ : Quiver V
        r : V
        height : V → Nat
        height_lt : ∀ ⦃a b : V⦄, Quiver.Hom a b → LT.lt (height a) (height b)
        unique_arrow : ∀ ⦃a b c : V⦄ (e : Quiver.Hom a c) (f : Quiver.Hom b c), And (E …
        root_or_arrow : ∀ (b : V), Or (Eq b r) (Exists fun a => Nonempty (Quiver.Hom a …
        b : V
        height_le : ∀ {a b : V}, Quiver.Path a b → LE.le (height a) (height b)
        ⊢ ∀ (p q : Quiver.Path r b), Eq p q
      -/
      intro p q
      induction p with
      | nil =>
        rcases q with _ | ⟨q, f⟩
        · rfl
        · exact False.elim (lt_irrefl _ (lt_of_le_of_lt (height_le q) (height_lt f)))
      | cons p e ih =>
        rcases q with _ | ⟨q, f⟩
        · exact False.elim (lt_irrefl _ (lt_of_le_of_lt (height_le p) (height_lt e)))
        · rcases unique_arrow e f with ⟨⟨⟩, ⟨⟩⟩
          rw [ih]⟩


/-- `RootedConnected r` means that there is a path from `r` to any other vertex. -/
class RootedConnected {V : Type u} [Quiver V] (r : V) : Prop where
  nonempty_path : ∀ b : V, Nonempty (Path r b)


/-- A path from `r` of minimal length. -/
noncomputable def shortestPath (b : V) : Path r b :=
  WellFounded.min (measure Path.length).wf Set.univ Set.univ_nonempty


/-- The length of a path is at least the length of the shortest path -/
theorem shortest_path_spec {a : V} (p : Path r a) : (shortestPath r a).length ≤ p.length :=
  not_lt.mp (WellFounded.not_lt_min (measure _).wf Set.univ _ trivial)


/-- A subquiver which by construction is an arborescence. -/
def geodesicSubtree : WideSubquiver V := fun a b =>
  { e | ∃ p : Path r a, shortestPath r b = p.cons e }


noncomputable instance geodesicArborescence : Arborescence (geodesicSubtree r) :=
  arborescenceMk r (fun a => (shortestPath r a).length)
    (by
      /-
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        ⊢ ∀ ⦃a b : WideSubquiver.toType V (Quiver.geodesicSubtree r)⦄, Quiver.Hom a b  …
      -/
      rintro a b ⟨e, p, h⟩
      /-
        case mk.intro
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        a b : WideSubquiver.toType V (Quiver.geodesicSubtree r)
        e : Quiver.Hom a b
        p : Quiver.Path r a
        h : Eq (Quiver.shortestPath r b) (p.cons e)
        ⊢ LT.lt ((fun a => (Quiver.shortestPath r a).length) a) ((fun a => (Quiver.sho …
      -/
      simp_rw [h, Path.length_cons, Nat.lt_succ_iff]
      /-
        case mk.intro
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        a b : WideSubquiver.toType V (Quiver.geodesicSubtree r)
        e : Quiver.Hom a b
        p : Quiver.Path r a
        h : Eq (Quiver.shortestPath r b) (p.cons e)
        ⊢ LE.le (Quiver.shortestPath r a).length p.length
      -/
      apply shortest_path_spec)
      /-
        🎉 no goals
      -/
    (by
      /-
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        ⊢ ∀ ⦃a b c : WideSubquiver.toType V (Quiver.geodesicSubtree r)⦄ (e : Quiver.Ho …
      -/
      rintro a b c ⟨e, p, h⟩ ⟨f, q, j⟩
      /-
        case mk.intro.mk.intro
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        a b c : WideSubquiver.toType V (Quiver.geodesicSubtree r)
        e : Quiver.Hom a c
        p : Quiver.Path r a
        h : Eq (Quiver.shortestPath r c) (p.cons e)
        f : Quiver.Hom b c
        q : Quiver.Path r b
        j : Eq (Quiver.shortestPath r c) (q.cons f)
        ⊢ And (Eq a b) (HEq ⟨e, ⋯⟩ ⟨f, ⋯⟩)
      -/
      cases h.symm.trans j
      /-
        case mk.intro.mk.intro.refl
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        a c : WideSubquiver.toType V (Quiver.geodesicSubtree r)
        e : Quiver.Hom a c
        p : Quiver.Path r a
        h j : Eq (Quiver.shortestPath r c) (p.cons e)
        ⊢ And (Eq a a) (HEq ⟨e, ⋯⟩ ⟨e, ⋯⟩)
      -/
                      /-
                        🎉 no goals
                      -/
      constructor <;> rfl)
                      /-
                        🎉 no goals
                      -/
    (by
      /-
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        ⊢ ∀ (b : WideSubquiver.toType V (Quiver.geodesicSubtree r)), Or (Eq b r) (Exis …
      -/
      intro b
      /-
        V : Type u
        inst✝¹ : Quiver V
        r : V
        inst✝ : Quiver.RootedConnected r
        b : WideSubquiver.toType V (Quiver.geodesicSubtree r)
        ⊢ Or (Eq b r) (Exists fun a => Nonempty (Quiver.Hom a b))
      -/
      rcases hp : shortestPath r b with (_ | ⟨p, e⟩)
        /-
          case nil
          V : Type u
          inst✝¹ : Quiver V
          r : V
          inst✝ : Quiver.RootedConnected r
          hp : Eq (Quiver.shortestPath r r) Quiver.Path.nil
          ⊢ Or (Eq r r) (Exists fun a => Nonempty (Quiver.Hom a r))
        -/
      · exact Or.inl rfl
        /-
          🎉 no goals
        -/
        /-
          case cons
          V : Type u
          inst✝¹ : Quiver V
          r : V
          inst✝ : Quiver.RootedConnected r
          b : WideSubquiver.toType V (Quiver.geodesicSubtree r)
          b✝ : V
          p : Quiver.Path r b✝
          e : Quiver.Hom b✝ b
          hp : Eq (Quiver.shortestPath r b) (p.cons e)
          ⊢ Or (Eq b r) (Exists fun a => Nonempty (Quiver.Hom a b))
        -/
      · exact Or.inr ⟨_, ⟨⟨e, p, hp⟩⟩⟩)
        /-
          🎉 no goals
        -/


