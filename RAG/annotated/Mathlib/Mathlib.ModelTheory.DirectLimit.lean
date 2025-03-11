alias map_self := DirectedSystem.map_self'

alias map_map := DirectedSystem.map_map'


/-- Given a chain of embeddings of structures indexed by `ℕ`, defines a `DirectedSystem` by
composing them. -/
def natLERec (m n : ℕ) (h : m ≤ n) : G' m ↪[L] G' n :=
  Nat.leRecOn h (@fun k g => (f' k).comp g) (Embedding.refl L _)


@[simp]
theorem coe_natLERec (m n : ℕ) (h : m ≤ n) :
    (natLERec f' m n h : G' m → G' n) = Nat.leRecOn h (@fun k => f' k) := by
  /-
    L : FirstOrder.Language
    G' : Nat → Type w
    inst✝ : (i : Nat) → L.Structure (G' i)
    f' : (n : Nat) → L.Embedding (G' n) (G' (HAdd.hAdd n 1))
    m n : Nat
    h : LE.le m n
    ⊢ Eq ⇑(FirstOrder.Language.DirectedSystem.natLERec f' m n h) fun a => Nat.leRe …
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  /-
    case intro
    L : FirstOrder.Language
    G' : Nat → Type w
    inst✝ : (i : Nat) → L.Structure (G' i)
    f' : (n : Nat) → L.Embedding (G' n) (G' (HAdd.hAdd n 1))
    m k : Nat
    h : LE.le m (HAdd.hAdd m k)
    ⊢ Eq ⇑(FirstOrder.Language.DirectedSystem.natLERec f' m (HAdd.hAdd m k) h) fun …
  -/
  ext x
  /-
    case intro.h
    L : FirstOrder.Language
    G' : Nat → Type w
    inst✝ : (i : Nat) → L.Structure (G' i)
    f' : (n : Nat) → L.Embedding (G' n) (G' (HAdd.hAdd n 1))
    m k : Nat
    h : LE.le m (HAdd.hAdd m k)
    x : G' m
    ⊢ Eq ((FirstOrder.Language.DirectedSystem.natLERec f' m (HAdd.hAdd m k) h) x)  …
  -/
  induction' k with k ih
  · -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case intro.h.zero
      L : FirstOrder.Language
      G' : Nat → Type w
      inst✝ : (i : Nat) → L.Structure (G' i)
      f' : (n : Nat) → L.Embedding (G' n) (G' (HAdd.hAdd n 1))
      m : Nat
      x : G' m
      h : LE.le m (HAdd.hAdd m 0)
      ⊢ Eq ((FirstOrder.Language.DirectedSystem.natLERec f' m (HAdd.hAdd m 0) h) x)  …
    -/
    erw [natLERec, Nat.leRecOn_self, Embedding.refl_apply, Nat.leRecOn_self]
    /-
      🎉 no goals
    -/
  · -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    erw [Nat.leRecOn_succ le_self_add, natLERec, Nat.leRecOn_succ le_self_add, ← natLERec,
      Embedding.comp_apply, ih]


instance natLERec.directedSystem : DirectedSystem G' fun i j h => natLERec f' i j h :=
  ⟨fun _ _ => congr (congr rfl (Nat.leRecOn_self _)) rfl,
                           /-
                             L : FirstOrder.Language
                             ι : Type v
                             inst✝² : Preorder ι
                             G : ι → Type w
                             inst✝¹ : (i : ι) → L.Structure (G i)
                             f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
                             G' : Nat → Type w
                             inst✝ : (i : Nat) → L.Structure (G' i)
                             f' : (n : Nat) → L.Embedding (G' n) (G' (HAdd.hAdd n 1))
                             x✝² x✝¹ x✝ : Nat
                             hij : LE.le x✝ x✝¹
                             hjk : LE.le x✝¹ x✝²
                             ⊢ ∀ (x : G' x✝), Eq ((FirstOrder.Language.DirectedSystem.natLERec f' x✝¹ x✝² h …
                           -/
   fun _ _ _ hij hjk => by simp [Nat.leRecOn_trans hij hjk]⟩
                           /-
                             🎉 no goals
                           -/


set_option linter.unusedVariables false in
/-- Alias for `Σ i, G i`. -/
@[nolint unusedArguments]
protected abbrev Structure.Sigma (f : ∀ i j, i ≤ j → G i ↪[L] G j) := Σ i, G i

-- Porting note: Setting up notation for `Language.Structure.Sigma`: add a little asterisk to `Σ`

local notation "Σˣ" => Structure.Sigma


/-- Constructor for `FirstOrder.Language.Structure.Sigma` alias. -/
abbrev Structure.Sigma.mk (i : ι) (x : G i) : Σˣ f := ⟨i, x⟩


/-- Raises a family of elements in the `Σ`-type to the same level along the embeddings. -/
def unify {α : Type*} (x : α → Σˣ f) (i : ι) (h : i ∈ upperBounds (range (Sigma.fst ∘ x)))
    (a : α) : G i :=
  f (x a).1 i (h (mem_range_self a)) (x a).2


@[simp]
theorem unify_sigma_mk_self {α : Type*} {i : ι} {x : α → G i} :
    (unify f (fun a => .mk f i (x a)) i fun _ ⟨_, hj⟩ =>
      _root_.trans (le_of_eq hj.symm) (refl _)) = x := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝² : Preorder ι
    G : ι → Type w
    inst✝¹ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    α : Type u_1
    i : ι
    x : α → G i
    ⊢ Eq (FirstOrder.Language.DirectLimit.unify f (fun a => FirstOrder.Language.St …
  -/
  ext a
  /-
    case h
    L : FirstOrder.Language
    ι : Type v
    inst✝² : Preorder ι
    G : ι → Type w
    inst✝¹ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    α : Type u_1
    i : ι
    x : α → G i
    a : α
    ⊢ Eq (FirstOrder.Language.DirectLimit.unify f (fun a => FirstOrder.Language.St …
  -/
  rw [unify]
  /-
    case h
    L : FirstOrder.Language
    ι : Type v
    inst✝² : Preorder ι
    G : ι → Type w
    inst✝¹ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    α : Type u_1
    i : ι
    x : α → G i
    a : α
    ⊢ Eq ((f (FirstOrder.Language.Structure.Sigma.mk f i (x a)).fst i ⋯) (FirstOrd …
  -/
  apply DirectedSystem.map_self
  /-
    🎉 no goals
  -/


theorem comp_unify {α : Type*} {x : α → Σˣ f} {i j : ι} (ij : i ≤ j)
    (h : i ∈ upperBounds (range (Sigma.fst ∘ x))) :
    f i j ij ∘ unify f x i h = unify f x j
      fun k hk => _root_.trans (mem_upperBounds.1 h k hk) ij := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝² : Preorder ι
    G : ι → Type w
    inst✝¹ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    α : Type u_1
    x : α → FirstOrder.Language.Structure.Sigma f
    i j : ι
    ij : LE.le i j
    h : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    ⊢ Eq (Function.comp (⇑(f i j ij)) (FirstOrder.Language.DirectLimit.unify f x i …
  -/
  ext a
  /-
    case h
    L : FirstOrder.Language
    ι : Type v
    inst✝² : Preorder ι
    G : ι → Type w
    inst✝¹ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    α : Type u_1
    x : α → FirstOrder.Language.Structure.Sigma f
    i j : ι
    ij : LE.le i j
    h : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    a : α
    ⊢ Eq (Function.comp (⇑(f i j ij)) (FirstOrder.Language.DirectLimit.unify f x i …
  -/
  simp [unify, DirectedSystem.map_map]
  /-
    🎉 no goals
  -/


/-- The directed limit glues together the structures along the embeddings. -/
def setoid [DirectedSystem G fun i j h => f i j h] [IsDirected ι (· ≤ ·)] : Setoid (Σˣ f) where
  r := fun ⟨i, x⟩ ⟨j, y⟩ => ∃ (k : ι) (ik : i ≤ k) (jk : j ≤ k), f i k ik x = f j k jk y
  iseqv :=
    ⟨fun ⟨i, _⟩ => ⟨i, refl i, refl i, rfl⟩, @fun ⟨_, _⟩ ⟨_, _⟩ ⟨k, ik, jk, h⟩ =>
      ⟨k, jk, ik, h.symm⟩,
      @fun ⟨i, x⟩ ⟨j, y⟩ ⟨k, z⟩ ⟨ij, hiij, hjij, hij⟩ ⟨jk, hjjk, hkjk, hjk⟩ => by
        /-
          L : FirstOrder.Language
          ι : Type v
          inst✝³ : Preorder ι
          G : ι → Type w
          inst✝² : (i : ι) → L.Structure (G i)
          f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
          i : ι
          x : G i
          j : ι
          y : G j
          x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
          k : ι
          z : G k
          x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
          ij : ι
          hiij : LE.le i ij
          hjij : LE.le j ij
          hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
          jk : ι
          hjjk : LE.le j jk
          hkjk : LE.le k jk
          hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
          ⊢ FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x⟩ fu …
        -/
        obtain ⟨ijk, hijijk, hjkijk⟩ := directed_of (· ≤ ·) ij jk
        /-
          case intro.intro
          L : FirstOrder.Language
          ι : Type v
          inst✝³ : Preorder ι
          G : ι → Type w
          inst✝² : (i : ι) → L.Structure (G i)
          f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
          i : ι
          x : G i
          j : ι
          y : G j
          x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
          k : ι
          z : G k
          x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
          ij : ι
          hiij : LE.le i ij
          hjij : LE.le j ij
          hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
          jk : ι
          hjjk : LE.le j jk
          hkjk : LE.le k jk
          hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
          ijk : ι
          hijijk : LE.le ij ijk
          hjkijk : LE.le jk ijk
          ⊢ FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x⟩ fu …
        -/
        refine ⟨ijk, le_trans hiij hijijk, le_trans hkjk hjkijk, ?_⟩
        /-
          case intro.intro
          L : FirstOrder.Language
          ι : Type v
          inst✝³ : Preorder ι
          G : ι → Type w
          inst✝² : (i : ι) → L.Structure (G i)
          f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
          i : ι
          x : G i
          j : ι
          y : G j
          x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
          k : ι
          z : G k
          x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
          ij : ι
          hiij : LE.le i ij
          hjij : LE.le j ij
          hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
          jk : ι
          hjjk : LE.le j jk
          hkjk : LE.le k jk
          hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
          ijk : ι
          hijijk : LE.le ij ijk
          hjkijk : LE.le jk ijk
          ⊢ Eq ((f i ijk ⋯) x) ((f k ijk ⋯) z)
        -/
        rw [← DirectedSystem.map_map, hij, DirectedSystem.map_map]
          /-
            case intro.intro
            L : FirstOrder.Language
            ι : Type v
            inst✝³ : Preorder ι
            G : ι → Type w
            inst✝² : (i : ι) → L.Structure (G i)
            f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
            inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
            inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
            x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
            i : ι
            x : G i
            j : ι
            y : G j
            x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
            k : ι
            z : G k
            x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
            ij : ι
            hiij : LE.le i ij
            hjij : LE.le j ij
            hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
            jk : ι
            hjjk : LE.le j jk
            hkjk : LE.le k jk
            hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
            ijk : ι
            hijijk : LE.le ij ijk
            hjkijk : LE.le jk ijk
            ⊢ Eq ((f j ijk ⋯) y) ((f k ijk ⋯) z)
          -/
        · symm
          /-
            case intro.intro
            L : FirstOrder.Language
            ι : Type v
            inst✝³ : Preorder ι
            G : ι → Type w
            inst✝² : (i : ι) → L.Structure (G i)
            f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
            inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
            inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
            x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
            i : ι
            x : G i
            j : ι
            y : G j
            x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
            k : ι
            z : G k
            x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
            ij : ι
            hiij : LE.le i ij
            hjij : LE.le j ij
            hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
            jk : ι
            hjjk : LE.le j jk
            hkjk : LE.le k jk
            hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
            ijk : ι
            hijijk : LE.le ij ijk
            hjkijk : LE.le jk ijk
            ⊢ Eq ((f k ijk ⋯) z) ((f j ijk ⋯) y)
          -/
          rw [← DirectedSystem.map_map, ← hjk, DirectedSystem.map_map]
          /-
            case intro.intro.hjk
            L : FirstOrder.Language
            ι : Type v
            inst✝³ : Preorder ι
            G : ι → Type w
            inst✝² : (i : ι) → L.Structure (G i)
            f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
            inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
            inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
            x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
            i : ι
            x : G i
            j : ι
            y : G j
            x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
            k : ι
            z : G k
            x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
            ij : ι
            hiij : LE.le i ij
            hjij : LE.le j ij
            hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
            jk : ι
            hjjk : LE.le j jk
            hkjk : LE.le k jk
            hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
            ijk : ι
            hijijk : LE.le ij ijk
            hjkijk : LE.le jk ijk
            ⊢ LE.le jk ijk
          -/
          assumption
          /-
            🎉 no goals
          -/
        /-
          case intro.intro.hjk
          L : FirstOrder.Language
          ι : Type v
          inst✝³ : Preorder ι
          G : ι → Type w
          inst✝² : (i : ι) → L.Structure (G i)
          f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
          inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
          x✝⁴ x✝³ x✝² : FirstOrder.Language.Structure.Sigma f
          i : ι
          x : G i
          j : ι
          y : G j
          x✝¹ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨i, x …
          k : ι
          z : G k
          x✝ : FirstOrder.Language.DirectLimit.setoid.match_1 G f (fun x => Prop) ⟨j, y⟩ …
          ij : ι
          hiij : LE.le i ij
          hjij : LE.le j ij
          hij : Eq ((f i ij hiij) x) ((f j ij hjij) y)
          jk : ι
          hjjk : LE.le j jk
          hkjk : LE.le k jk
          hjk : Eq ((f j jk hjjk) y) ((f k jk hkjk) z)
          ijk : ι
          hijijk : LE.le ij ijk
          hjkijk : LE.le jk ijk
          ⊢ LE.le ij ijk
        -/
        assumption⟩
        /-
          🎉 no goals
        -/


/-- The structure on the `Σ`-type which becomes the structure on the direct limit after quotienting.
 -/
noncomputable def sigmaStructure [IsDirected ι (· ≤ ·)] [Nonempty ι] : L.Structure (Σˣ f) where
  funMap F x :=
    ⟨_,
      funMap F
        (unify f x (Classical.choose (Finite.bddAbove_range fun a => (x a).1))
          (Classical.choose_spec (Finite.bddAbove_range fun a => (x a).1)))⟩
  RelMap R x :=
    RelMap R
      (unify f x (Classical.choose (Finite.bddAbove_range fun a => (x a).1))
        (Classical.choose_spec (Finite.bddAbove_range fun a => (x a).1)))


/-- The direct limit of a directed system is the structures glued together along the embeddings. -/
def DirectLimit [DirectedSystem G fun i j h => f i j h] [IsDirected ι (· ≤ ·)] :=
  Quotient (DirectLimit.setoid G f)


instance [DirectedSystem G fun i j h => f i j h] [IsDirected ι (· ≤ ·)] [Inhabited ι]
    [Inhabited (G default)] : Inhabited (DirectLimit G f) :=
  ⟨⟦⟨default, default⟩⟧⟩


theorem equiv_iff {x y : Σˣ f} {i : ι} (hx : x.1 ≤ i) (hy : y.1 ≤ i) :
    x ≈ y ↔ (f x.1 i hx) x.2 = (f y.1 i hy) y.2 := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    x y : FirstOrder.Language.Structure.Sigma f
    i : ι
    hx : LE.le x.fst i
    hy : LE.le y.fst i
    ⊢ Iff (HasEquiv.Equiv x y) (Eq ((f x.fst i hx) x.snd) ((f y.fst i hy) y.snd))
  -/
  cases x
  /-
    case mk
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    y : FirstOrder.Language.Structure.Sigma f
    i : ι
    hy : LE.le y.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hx : LE.le ⟨fst✝, snd✝⟩.fst i
    ⊢ Iff (HasEquiv.Equiv ⟨fst✝, snd✝⟩ y) (Eq ((f ⟨fst✝, snd✝⟩.fst i hx) ⟨fst✝, sn …
  -/
  cases y
  /-
    case mk.mk
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    ⊢ Iff (HasEquiv.Equiv ⟨fst✝¹, snd✝¹⟩ ⟨fst✝, snd✝⟩) (Eq ((f ⟨fst✝¹, snd✝¹⟩.fst  …
  -/
  refine ⟨fun xy => ?_, fun xy => ⟨i, hx, hy, xy⟩⟩
  /-
    case mk.mk
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    xy : HasEquiv.Equiv ⟨fst✝¹, snd✝¹⟩ ⟨fst✝, snd✝⟩
    ⊢ Eq ((f ⟨fst✝¹, snd✝¹⟩.fst i hx) ⟨fst✝¹, snd✝¹⟩.snd) ((f ⟨fst✝, snd✝⟩.fst i h …
  -/
  obtain ⟨j, _, _, h⟩ := xy
  /-
    case mk.mk.intro.intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    j : ι
    w✝¹ : LE.le fst✝¹ j
    w✝ : LE.le fst✝ j
    h : Eq ((f fst✝¹ j w✝¹) snd✝¹) ((f fst✝ j w✝) snd✝)
    ⊢ Eq ((f ⟨fst✝¹, snd✝¹⟩.fst i hx) ⟨fst✝¹, snd✝¹⟩.snd) ((f ⟨fst✝, snd✝⟩.fst i h …
  -/
  obtain ⟨k, ik, jk⟩ := directed_of (· ≤ ·) i j
  /-
    case mk.mk.intro.intro.intro.intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    j : ι
    w✝¹ : LE.le fst✝¹ j
    w✝ : LE.le fst✝ j
    h : Eq ((f fst✝¹ j w✝¹) snd✝¹) ((f fst✝ j w✝) snd✝)
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    ⊢ Eq ((f ⟨fst✝¹, snd✝¹⟩.fst i hx) ⟨fst✝¹, snd✝¹⟩.snd) ((f ⟨fst✝, snd✝⟩.fst i h …
  -/
  have h := congr_arg (f j k jk) h
  /-
    case mk.mk.intro.intro.intro.intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    j : ι
    w✝¹ : LE.le fst✝¹ j
    w✝ : LE.le fst✝ j
    h✝ : Eq ((f fst✝¹ j w✝¹) snd✝¹) ((f fst✝ j w✝) snd✝)
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    h : Eq ((f j k jk) ((f fst✝¹ j w✝¹) snd✝¹)) ((f j k jk) ((f fst✝ j w✝) snd✝))
    ⊢ Eq ((f ⟨fst✝¹, snd✝¹⟩.fst i hx) ⟨fst✝¹, snd✝¹⟩.snd) ((f ⟨fst✝, snd✝⟩.fst i h …
  -/
  apply (f i k ik).injective
  /-
    case mk.mk.intro.intro.intro.intro.intro.a
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    j : ι
    w✝¹ : LE.le fst✝¹ j
    w✝ : LE.le fst✝ j
    h✝ : Eq ((f fst✝¹ j w✝¹) snd✝¹) ((f fst✝ j w✝) snd✝)
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    h : Eq ((f j k jk) ((f fst✝¹ j w✝¹) snd✝¹)) ((f j k jk) ((f fst✝ j w✝) snd✝))
    ⊢ Eq ((f i k ik) ((f ⟨fst✝¹, snd✝¹⟩.fst i hx) ⟨fst✝¹, snd✝¹⟩.snd)) ((f i k ik) …
  -/
  rw [DirectedSystem.map_map, DirectedSystem.map_map] at *
  /-
    case mk.mk.intro.intro.intro.intro.intro.a
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    i fst✝¹ : ι
    snd✝¹ : G fst✝¹
    hx : LE.le ⟨fst✝¹, snd✝¹⟩.fst i
    fst✝ : ι
    snd✝ : G fst✝
    hy : LE.le ⟨fst✝, snd✝⟩.fst i
    j : ι
    w✝¹ : LE.le fst✝¹ j
    w✝ : LE.le fst✝ j
    h✝ : Eq ((f fst✝¹ j w✝¹) snd✝¹) ((f fst✝ j w✝) snd✝)
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    h : Eq ((f fst✝¹ k ⋯) snd✝¹) ((f fst✝ k ⋯) snd✝)
    ⊢ Eq ((f ⟨fst✝¹, snd✝¹⟩.fst k ⋯) ⟨fst✝¹, snd✝¹⟩.snd) ((f ⟨fst✝, snd✝⟩.fst k ⋯) …
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem funMap_unify_equiv {n : ℕ} (F : L.Functions n) (x : Fin n → Σˣ f) (i j : ι)
    (hi : i ∈ upperBounds (range (Sigma.fst ∘ x))) (hj : j ∈ upperBounds (range (Sigma.fst ∘ x))) :
    Structure.Sigma.mk f i (funMap F (unify f x i hi)) ≈ .mk f j (funMap F (unify f x j hj)) := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    n : Nat
    F : L.Functions n
    x : Fin n → FirstOrder.Language.Structure.Sigma f
    i j : ι
    hi : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    hj : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) j
    ⊢ HasEquiv.Equiv (FirstOrder.Language.Structure.Sigma.mk f i (FirstOrder.Langu …
  -/
  obtain ⟨k, ik, jk⟩ := directed_of (· ≤ ·) i j
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    n : Nat
    F : L.Functions n
    x : Fin n → FirstOrder.Language.Structure.Sigma f
    i j : ι
    hi : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    hj : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) j
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    ⊢ HasEquiv.Equiv (FirstOrder.Language.Structure.Sigma.mk f i (FirstOrder.Langu …
  -/
  refine ⟨k, ik, jk, ?_⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    n : Nat
    F : L.Functions n
    x : Fin n → FirstOrder.Language.Structure.Sigma f
    i j : ι
    hi : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    hj : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) j
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    ⊢ Eq ((f i k ik) (FirstOrder.Language.Structure.funMap F (FirstOrder.Language. …
  -/
  rw [(f i k ik).map_fun, (f j k jk).map_fun, comp_unify, comp_unify]
  /-
    🎉 no goals
  -/


theorem relMap_unify_equiv {n : ℕ} (R : L.Relations n) (x : Fin n → Σˣ f) (i j : ι)
    (hi : i ∈ upperBounds (range (Sigma.fst ∘ x))) (hj : j ∈ upperBounds (range (Sigma.fst ∘ x))) :
    RelMap R (unify f x i hi) = RelMap R (unify f x j hj) := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    n : Nat
    R : L.Relations n
    x : Fin n → FirstOrder.Language.Structure.Sigma f
    i j : ι
    hi : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    hj : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) j
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R (FirstOrder.Language.DirectLimit. …
  -/
  obtain ⟨k, ik, jk⟩ := directed_of (· ≤ ·) i j
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    G : ι → Type w
    inst✝² : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    n : Nat
    R : L.Relations n
    x : Fin n → FirstOrder.Language.Structure.Sigma f
    i j : ι
    hi : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
    hj : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) j
    k : ι
    ik : LE.le i k
    jk : LE.le j k
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R (FirstOrder.Language.DirectLimit. …
  -/
  rw [← (f i k ik).map_rel, comp_unify, ← (f j k jk).map_rel, comp_unify]
  /-
    🎉 no goals
  -/


theorem exists_unify_eq {α : Type*} [Finite α] {x y : α → Σˣ f} (xy : x ≈ y) :
    ∃ (i : ι) (hx : i ∈ upperBounds (range (Sigma.fst ∘ x)))
      (hy : i ∈ upperBounds (range (Sigma.fst ∘ y))), unify f x i hx = unify f y i hy := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x y : α → FirstOrder.Language.Structure.Sigma f
    xy : HasEquiv.Equiv x y
    ⊢ Exists fun i => Exists fun hx => Exists fun hy => Eq (FirstOrder.Language.Di …
  -/
  obtain ⟨i, hi⟩ := Finite.bddAbove_range (Sum.elim (fun a => (x a).1) fun a => (y a).1)
  /-
    case intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x y : α → FirstOrder.Language.Structure.Sigma f
    xy : HasEquiv.Equiv x y
    i : ι
    hi : Membership.mem (upperBounds (Set.range (Sum.elim (fun a => (x a).fst) fun …
    ⊢ Exists fun i => Exists fun hx => Exists fun hy => Eq (FirstOrder.Language.Di …
  -/
  rw [Sum.elim_range, upperBounds_union] at hi
  /-
    case intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x y : α → FirstOrder.Language.Structure.Sigma f
    xy : HasEquiv.Equiv x y
    i : ι
    hi : Membership.mem (Inter.inter (upperBounds (Set.range fun a => (x a).fst))  …
    ⊢ Exists fun i => Exists fun hx => Exists fun hy => Eq (FirstOrder.Language.Di …
  -/
  simp_rw [← Function.comp_apply (f := Sigma.fst)] at hi
  /-
    case intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x y : α → FirstOrder.Language.Structure.Sigma f
    xy : HasEquiv.Equiv x y
    i : ι
    hi : Membership.mem (Inter.inter (upperBounds (Set.range fun a => Function.com …
    ⊢ Exists fun i => Exists fun hx => Exists fun hy => Eq (FirstOrder.Language.Di …
  -/
  exact ⟨i, hi.1, hi.2, funext fun a => (equiv_iff G f _ _).1 (xy a)⟩
  /-
    🎉 no goals
  -/


theorem funMap_equiv_unify {n : ℕ} (F : L.Functions n) (x : Fin n → Σˣ f) (i : ι)
    (hi : i ∈ upperBounds (range (Sigma.fst ∘ x))) :
    funMap F x ≈ .mk f _ (funMap F (unify f x i hi)) :=
  funMap_unify_equiv G f F x (Classical.choose (Finite.bddAbove_range fun a => (x a).1)) i _ hi


theorem relMap_equiv_unify {n : ℕ} (R : L.Relations n) (x : Fin n → Σˣ f) (i : ι)
    (hi : i ∈ upperBounds (range (Sigma.fst ∘ x))) :
    RelMap R x = RelMap R (unify f x i hi) :=
  relMap_unify_equiv G f R x (Classical.choose (Finite.bddAbove_range fun a => (x a).1)) i _ hi


/-- The direct limit `setoid` respects the structure `sigmaStructure`, so quotienting by it
  gives rise to a valid structure. -/
noncomputable instance prestructure : L.Prestructure (DirectLimit.setoid G f) where
  toStructure := sigmaStructure G f
  fun_equiv {n} {F} x y xy := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      n : Nat
      F : L.Functions n
      x y : Fin n → FirstOrder.Language.Structure.Sigma f
      xy : HasEquiv.Equiv x y
      ⊢ HasEquiv.Equiv (FirstOrder.Language.Structure.funMap F x) (FirstOrder.Langua …
    -/
    obtain ⟨i, hx, hy, h⟩ := exists_unify_eq G f xy
    refine
      Setoid.trans (funMap_equiv_unify G f F x i hx)
        (Setoid.trans ?_ (Setoid.symm (funMap_equiv_unify G f F y i hy)))
    /-
      case intro.intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      n : Nat
      F : L.Functions n
      x y : Fin n → FirstOrder.Language.Structure.Sigma f
      xy : HasEquiv.Equiv x y
      i : ι
      hx : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
      hy : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst y))) i
      h : Eq (FirstOrder.Language.DirectLimit.unify f x i hx) (FirstOrder.Language.D …
      ⊢ HasEquiv.Equiv (FirstOrder.Language.Structure.Sigma.mk f i (FirstOrder.Langu …
    -/
    rw [h]
    /-
      🎉 no goals
    -/
  rel_equiv {n} {R} x y xy := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      n : Nat
      R : L.Relations n
      x y : Fin n → FirstOrder.Language.Structure.Sigma f
      xy : HasEquiv.Equiv x y
      ⊢ Eq (FirstOrder.Language.Structure.RelMap R x) (FirstOrder.Language.Structure …
    -/
    obtain ⟨i, hx, hy, h⟩ := exists_unify_eq G f xy
    refine _root_.trans (relMap_equiv_unify G f R x i hx)
      (_root_.trans ?_ (symm (relMap_equiv_unify G f R y i hy)))
    /-
      case intro.intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      n : Nat
      R : L.Relations n
      x y : Fin n → FirstOrder.Language.Structure.Sigma f
      xy : HasEquiv.Equiv x y
      i : ι
      hx : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst x))) i
      hy : Membership.mem (upperBounds (Set.range (Function.comp Sigma.fst y))) i
      h : Eq (FirstOrder.Language.DirectLimit.unify f x i hx) (FirstOrder.Language.D …
      ⊢ Eq (FirstOrder.Language.Structure.RelMap R (FirstOrder.Language.DirectLimit. …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


/-- The `L.Structure` on a direct limit of `L.Structure`s. -/
noncomputable instance instStructureDirectLimit : L.Structure (DirectLimit G f) :=
  Language.quotientStructure


@[simp]
theorem funMap_quotient_mk'_sigma_mk' {n : ℕ} {F : L.Functions n} {i : ι} {x : Fin n → G i} :
    funMap F (fun a => (⟦.mk f i (x a)⟧ : DirectLimit G f)) = ⟦.mk f i (funMap F x)⟧ := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    F : L.Functions n
    i : ι
    x : Fin n → G i
    ⊢ Eq (FirstOrder.Language.Structure.funMap F fun a => Quotient.mk (FirstOrder. …
  -/
  simp only [funMap_quotient_mk', Quotient.eq]
  obtain ⟨k, ik, jk⟩ :=
    directed_of (· ≤ ·) i (Classical.choose (Finite.bddAbove_range fun _ : Fin n => i))
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    F : L.Functions n
    i : ι
    x : Fin n → G i
    k : ι
    ik : LE.le i k
    jk : LE.le (Classical.choose ⋯) k
    ⊢ (FirstOrder.Language.DirectLimit.setoid G f) (FirstOrder.Language.Structure. …
  -/
  refine ⟨k, jk, ik, ?_⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    F : L.Functions n
    i : ι
    x : Fin n → G i
    k : ι
    ik : LE.le i k
    jk : LE.le (Classical.choose ⋯) k
    ⊢ Eq ((f (Classical.choose ⋯) k jk) (FirstOrder.Language.Structure.funMap F (F …
  -/
  simp only [Embedding.map_fun, comp_unify]
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    F : L.Functions n
    i : ι
    x : Fin n → G i
    k : ι
    ik : LE.le i k
    jk : LE.le (Classical.choose ⋯) k
    ⊢ Eq (FirstOrder.Language.Structure.funMap F (FirstOrder.Language.DirectLimit. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem relMap_quotient_mk'_sigma_mk' {n : ℕ} {R : L.Relations n} {i : ι} {x : Fin n → G i} :
    RelMap R (fun a => (⟦.mk f i (x a)⟧ : DirectLimit G f)) = RelMap R x := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    R : L.Relations n
    i : ι
    x : Fin n → G i
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R fun a => Quotient.mk (FirstOrder. …
  -/
  rw [relMap_quotient_mk']
  obtain ⟨k, _, _⟩ :=
    directed_of (· ≤ ·) i (Classical.choose (Finite.bddAbove_range fun _ : Fin n => i))
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    R : L.Relations n
    i : ι
    x : Fin n → G i
    k : ι
    left✝ : LE.le i k
    right✝ : LE.le (Classical.choose ⋯) k
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R fun a => FirstOrder.Language.Stru …
  -/
  rw [relMap_equiv_unify G f R (fun a => .mk f i (x a)) i]
  /-
    case intro.intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    n : Nat
    R : L.Relations n
    i : ι
    x : Fin n → G i
    k : ι
    left✝ : LE.le i k
    right✝ : LE.le (Classical.choose ⋯) k
    ⊢ Eq (FirstOrder.Language.Structure.RelMap R (FirstOrder.Language.DirectLimit. …
  -/
  rw [unify_sigma_mk_self]
  /-
    🎉 no goals
  -/


theorem exists_quotient_mk'_sigma_mk'_eq {α : Type*} [Finite α] (x : α → DirectLimit G f) :
    ∃ (i : ι) (y : α → G i), x = fun a => ⟦.mk f i (y a)⟧ := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    ⊢ Exists fun i => Exists fun y => Eq x fun a => Quotient.mk (FirstOrder.Langua …
  -/
  obtain ⟨i, hi⟩ := Finite.bddAbove_range fun a => (x a).out.1
  /-
    case intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    ⊢ Exists fun i => Exists fun y => Eq x fun a => Quotient.mk (FirstOrder.Langua …
  -/
  refine ⟨i, unify f (Quotient.out ∘ x) i hi, ?_⟩
  /-
    case intro
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    ⊢ Eq x fun a => Quotient.mk (FirstOrder.Language.DirectLimit.setoid G f) (Firs …
  -/
  ext a
  /-
    case intro.h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    a : α
    ⊢ Eq (x a) (Quotient.mk (FirstOrder.Language.DirectLimit.setoid G f) (FirstOrd …
  -/
  rw [Quotient.eq_mk_iff_out, unify]
  /-
    case intro.h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    a : α
    ⊢ HasEquiv.Equiv (Quotient.out (x a)) (FirstOrder.Language.Structure.Sigma.mk  …
  -/
  generalize_proofs r
  /-
    case intro.h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    a : α
    r : LE.le (Function.comp Quotient.out x a).fst i
    ⊢ HasEquiv.Equiv (Quotient.out (x a)) (FirstOrder.Language.Structure.Sigma.mk  …
  -/
  change _ ≈ Structure.Sigma.mk f i (f (Quotient.out (x a)).fst i r (Quotient.out (x a)).snd)
  have : (.mk f i (f (Quotient.out (x a)).fst i r (Quotient.out (x a)).snd) : Σˣ f).fst ≤ i :=
    le_rfl
  /-
    case intro.h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    a : α
    r : LE.le (Function.comp Quotient.out x a).fst i
    this : LE.le (FirstOrder.Language.Structure.Sigma.mk f i ((f (Quotient.out (x  …
    ⊢ HasEquiv.Equiv (Quotient.out (x a)) (FirstOrder.Language.Structure.Sigma.mk  …
  -/
  rw [equiv_iff G f (i := i) (hi _) this]
    /-
      case intro.h
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      α : Type u_1
      inst✝ : Finite α
      x : α → FirstOrder.Language.DirectLimit G f
      i : ι
      hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
      a : α
      r : LE.le (Function.comp Quotient.out x a).fst i
      this : LE.le (FirstOrder.Language.Structure.Sigma.mk f i ((f (Quotient.out (x  …
      ⊢ Eq ((f (Quotient.out (x a)).fst i ⋯) (Quotient.out (x a)).snd) ((f (FirstOrd …
    -/
  · simp only [DirectedSystem.map_self]
    /-
      🎉 no goals
    -/
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    α : Type u_1
    inst✝ : Finite α
    x : α → FirstOrder.Language.DirectLimit G f
    i : ι
    hi : Membership.mem (upperBounds (Set.range fun a => (Quotient.out (x a)).fst) …
    a : α
    r : LE.le (Function.comp Quotient.out x a).fst i
    this : LE.le (FirstOrder.Language.Structure.Sigma.mk f i ((f (Quotient.out (x  …
    ⊢ Membership.mem (Set.range fun a => (Quotient.out (x a)).fst) (Quotient.out ( …
  -/
  exact ⟨a, rfl⟩
  /-
    🎉 no goals
  -/


/-- The canonical map from a component to the direct limit. -/
def of (i : ι) : G i ↪[L] DirectLimit G f where
  toFun := fun a => ⟦.mk f i a⟧
  inj' x y h := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      x y : G i
      h : Eq ((fun a => Quotient.mk (FirstOrder.Language.DirectLimit.setoid G f) (Fi …
      ⊢ Eq x y
    -/
    rw [Quotient.eq] at h
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      x y : G i
      h : (FirstOrder.Language.DirectLimit.setoid G f) (FirstOrder.Language.Structur …
      ⊢ Eq x y
    -/
    obtain ⟨j, h1, _, h3⟩ := h
    /-
      case intro.intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      x y : G i
      j : ι
      h1 w✝ : LE.le i j
      h3 : Eq ((f i j h1) x) ((f i j w✝) y)
      ⊢ Eq x y
    -/
    exact (f i j h1).injective h3
    /-
      🎉 no goals
    -/
  map_fun' F x := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      n✝ : Nat
      F : L.Functions n✝
      x : Fin n✝ → G i
      ⊢ Eq ({ toFun := fun a => Quotient.mk (FirstOrder.Language.DirectLimit.setoid  …
    -/
    simp only
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      n✝ : Nat
      F : L.Functions n✝
      x : Fin n✝ → G i
      ⊢ Eq (Quotient.mk (FirstOrder.Language.DirectLimit.setoid G f) (FirstOrder.Lan …
    -/
    rw [← funMap_quotient_mk'_sigma_mk']
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      n✝ : Nat
      F : L.Functions n✝
      x : Fin n✝ → G i
      ⊢ Eq (FirstOrder.Language.Structure.funMap F fun a => Quotient.mk (FirstOrder. …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_rel' := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      ⊢ ∀ {n : Nat} (r : L.Relations n) (x : Fin n → G i), Iff (FirstOrder.Language. …
    -/
    intro n R x
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      n : Nat
      R : L.Relations n
      x : Fin n → G i
      ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp { toFun := fun a  …
    -/
    change RelMap R (fun a => (⟦.mk f i (x a)⟧ : DirectLimit G f)) ↔ _
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁴ : Preorder ι
      G : ι → Type w
      inst✝³ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : Nonempty ι
      i : ι
      n : Nat
      R : L.Relations n
      x : Fin n → G i
      ⊢ Iff (FirstOrder.Language.Structure.RelMap R fun a => Quotient.mk (FirstOrder …
    -/
    simp only [relMap_quotient_mk'_sigma_mk']
    /-
      🎉 no goals
    -/




@[simp]
theorem of_apply {i : ι} {x : G i} : of L ι G f i x = ⟦.mk f i x⟧ :=
  rfl

-- Porting note: removed the `@[simp]`, it is not in simp-normal form, but the simp-normal version
-- of this theorem would not be useful.

theorem of_f {i j : ι} {hij : i ≤ j} {x : G i} : of L ι G f j (f i j hij x) = of L ι G f i x := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    i j : ι
    hij : LE.le i j
    x : G i
    ⊢ Eq ((FirstOrder.Language.DirectLimit.of L ι G f j) ((f i j hij) x)) ((FirstO …
  -/
  rw [of_apply, of_apply, Quotient.eq]
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    i j : ι
    hij : LE.le i j
    x : G i
    ⊢ (FirstOrder.Language.DirectLimit.setoid G f) (FirstOrder.Language.Structure. …
  -/
  refine Setoid.symm ⟨j, hij, refl j, ?_⟩
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    i j : ι
    hij : LE.le i j
    x : G i
    ⊢ Eq ((f i j hij) x) ((f j j ⋯) ((f i j hij) x))
  -/
  simp only [DirectedSystem.map_self]
  /-
    🎉 no goals
  -/


/-- Every element of the direct limit corresponds to some element in
some component of the directed system. -/
theorem exists_of (z : DirectLimit G f) : ∃ i x, of L ι G f i x = z :=
                        /-
                          L : FirstOrder.Language
                          ι : Type v
                          inst✝⁴ : Preorder ι
                          G : ι → Type w
                          inst✝³ : (i : ι) → L.Structure (G i)
                          f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
                          inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
                          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
                          inst✝ : Nonempty ι
                          z : FirstOrder.Language.DirectLimit G f
                          ⊢ Eq ((FirstOrder.Language.DirectLimit.of L ι G f (Quotient.out z).fst) (Quoti …
                        -/
  ⟨z.out.1, z.out.2, by simp⟩
                        /-
                          🎉 no goals
                        -/


@[elab_as_elim]
protected theorem inductionOn {C : DirectLimit G f → Prop} (z : DirectLimit G f)
    (ih : ∀ i x, C (of L ι G f i x)) : C z :=
  let ⟨i, x, h⟩ := exists_of z
  h ▸ ih i x


theorem iSup_range_of_eq_top : ⨆ i, (of L ι G f i).toHom.range = ⊤ :=
  eq_top_iff.2 (fun x _ ↦ DirectLimit.inductionOn x
    (fun i _ ↦ le_iSup (fun i ↦ Hom.range (Embedding.toHom (of L ι G f i))) i (mem_range_self _)))


/-- Every finitely generated substructure of the direct limit corresponds to some
substructure in some component of the directed system. -/
theorem exists_fg_substructure_in_Sigma (S : L.Substructure (DirectLimit G f)) (S_fg : S.FG) :
    ∃ i, ∃ T : L.Substructure (G i), T.map (of L ι G f i).toHom = S := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    S : L.Substructure (FirstOrder.Language.DirectLimit G f)
    S_fg : S.FG
    ⊢ Exists fun i => Exists fun T => Eq (FirstOrder.Language.Substructure.map (Fi …
  -/
  let ⟨A, A_closure⟩ := S_fg
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    S : L.Substructure (FirstOrder.Language.DirectLimit G f)
    S_fg : S.FG
    A : Finset (FirstOrder.Language.DirectLimit G f)
    A_closure : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑A) S
    ⊢ Exists fun i => Exists fun T => Eq (FirstOrder.Language.Substructure.map (Fi …
  -/
  let ⟨i, y, eq_y⟩ := exists_quotient_mk'_sigma_mk'_eq G _ (fun a : A ↦ a.1)
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    S : L.Substructure (FirstOrder.Language.DirectLimit G f)
    S_fg : S.FG
    A : Finset (FirstOrder.Language.DirectLimit G f)
    A_closure : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑A) S
    i : ι
    y : (Subtype fun x => Membership.mem A x) → G i
    eq_y : Eq (fun a => ↑a) fun a => Quotient.mk (FirstOrder.Language.DirectLimit. …
    ⊢ Exists fun i => Exists fun T => Eq (FirstOrder.Language.Substructure.map (Fi …
  -/
  use i
  /-
    case h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    S : L.Substructure (FirstOrder.Language.DirectLimit G f)
    S_fg : S.FG
    A : Finset (FirstOrder.Language.DirectLimit G f)
    A_closure : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑A) S
    i : ι
    y : (Subtype fun x => Membership.mem A x) → G i
    eq_y : Eq (fun a => ↑a) fun a => Quotient.mk (FirstOrder.Language.DirectLimit. …
    ⊢ Exists fun T => Eq (FirstOrder.Language.Substructure.map (FirstOrder.Languag …
  -/
  use Substructure.closure L (range y)
  /-
    case h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    S : L.Substructure (FirstOrder.Language.DirectLimit G f)
    S_fg : S.FG
    A : Finset (FirstOrder.Language.DirectLimit G f)
    A_closure : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑A) S
    i : ι
    y : (Subtype fun x => Membership.mem A x) → G i
    eq_y : Eq (fun a => ↑a) fun a => Quotient.mk (FirstOrder.Language.DirectLimit. …
    ⊢ Eq (FirstOrder.Language.Substructure.map (FirstOrder.Language.DirectLimit.of …
  -/
  rw [Substructure.map_closure]
  /-
    case h
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝ : Nonempty ι
    S : L.Substructure (FirstOrder.Language.DirectLimit G f)
    S_fg : S.FG
    A : Finset (FirstOrder.Language.DirectLimit G f)
    A_closure : Eq ((FirstOrder.Language.Substructure.closure L).toFun ↑A) S
    i : ι
    y : (Subtype fun x => Membership.mem A x) → G i
    eq_y : Eq (fun a => ↑a) fun a => Quotient.mk (FirstOrder.Language.DirectLimit. …
    ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.image (⇑(FirstOr …
  -/
  simp only [Embedding.coe_toHom, of_apply]
  rw [← image_univ, image_image, image_univ, ← eq_y,
    Subtype.range_coe_subtype, Finset.setOf_mem, A_closure]


variable (L ι G f) in
/-- The universal property of the direct limit: maps from the components to another module
that respect the directed system structure (i.e. make some diagram commute) give rise
to a unique map out of the direct limit. -/
def lift (g : ∀ i, G i ↪[L] P) (Hg : ∀ i j hij x, g j (f i j hij x) = g i x) :
    DirectLimit G f ↪[L] P where
  toFun :=
    Quotient.lift (fun x : Σˣ f => (g x.1) x.2) fun x y xy => by
      /-
        L : FirstOrder.Language
        ι : Type v
        inst✝⁵ : Preorder ι
        G : ι → Type w
        inst✝⁴ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝¹ : Nonempty ι
        P : Type u₁
        inst✝ : L.Structure P
        g : (i : ι) → L.Embedding (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
        x y : FirstOrder.Language.Structure.Sigma f
        xy : HasEquiv.Equiv x y
        ⊢ Eq ((fun x => (g x.fst) x.snd) x) ((fun x => (g x.fst) x.snd) y)
      -/
      simp only
      /-
        L : FirstOrder.Language
        ι : Type v
        inst✝⁵ : Preorder ι
        G : ι → Type w
        inst✝⁴ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝¹ : Nonempty ι
        P : Type u₁
        inst✝ : L.Structure P
        g : (i : ι) → L.Embedding (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
        x y : FirstOrder.Language.Structure.Sigma f
        xy : HasEquiv.Equiv x y
        ⊢ Eq ((g x.fst) x.snd) ((g y.fst) y.snd)
      -/
      obtain ⟨i, hx, hy⟩ := directed_of (· ≤ ·) x.1 y.1
      /-
        case intro.intro
        L : FirstOrder.Language
        ι : Type v
        inst✝⁵ : Preorder ι
        G : ι → Type w
        inst✝⁴ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝¹ : Nonempty ι
        P : Type u₁
        inst✝ : L.Structure P
        g : (i : ι) → L.Embedding (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
        x y : FirstOrder.Language.Structure.Sigma f
        xy : HasEquiv.Equiv x y
        i : ι
        hx : LE.le x.fst i
        hy : LE.le y.fst i
        ⊢ Eq ((g x.fst) x.snd) ((g y.fst) y.snd)
      -/
      rw [← Hg x.1 i hx, ← Hg y.1 i hy]
      /-
        case intro.intro
        L : FirstOrder.Language
        ι : Type v
        inst✝⁵ : Preorder ι
        G : ι → Type w
        inst✝⁴ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝¹ : Nonempty ι
        P : Type u₁
        inst✝ : L.Structure P
        g : (i : ι) → L.Embedding (G i) P
        Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
        x y : FirstOrder.Language.Structure.Sigma f
        xy : HasEquiv.Equiv x y
        i : ι
        hx : LE.le x.fst i
        hy : LE.le y.fst i
        ⊢ Eq ((g i) ((f x.fst i hx) x.snd)) ((g i) ((f y.fst i hy) y.snd))
      -/
      exact congr_arg _ ((equiv_iff ..).1 xy)
      /-
        🎉 no goals
      -/
  inj' x y xy := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      x y : FirstOrder.Language.DirectLimit G f
      xy : Eq (Quotient.lift (fun x => (g x.fst) x.snd) ⋯ x) (Quotient.lift (fun x = …
      ⊢ Eq x y
    -/
    rw [← Quotient.out_eq x, ← Quotient.out_eq y, Quotient.lift_mk, Quotient.lift_mk] at xy
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      x y : FirstOrder.Language.DirectLimit G f
      xy : Eq ((g (Quotient.out x).fst) (Quotient.out x).snd) ((g (Quotient.out y).f …
      ⊢ Eq x y
    -/
    obtain ⟨i, hx, hy⟩ := directed_of (· ≤ ·) x.out.1 y.out.1
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      x y : FirstOrder.Language.DirectLimit G f
      xy : Eq ((g (Quotient.out x).fst) (Quotient.out x).snd) ((g (Quotient.out y).f …
      i : ι
      hx : LE.le (Quotient.out x).fst i
      hy : LE.le (Quotient.out y).fst i
      ⊢ Eq x y
    -/
    rw [← Hg x.out.1 i hx, ← Hg y.out.1 i hy] at xy
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      x y : FirstOrder.Language.DirectLimit G f
      i : ι
      hx : LE.le (Quotient.out x).fst i
      hy : LE.le (Quotient.out y).fst i
      xy : Eq ((g i) ((f (Quotient.out x).fst i hx) (Quotient.out x).snd)) ((g i) (( …
      ⊢ Eq x y
    -/
    rw [← Quotient.out_eq x, ← Quotient.out_eq y, Quotient.eq_iff_equiv, equiv_iff G f hx hy]
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      x y : FirstOrder.Language.DirectLimit G f
      i : ι
      hx : LE.le (Quotient.out x).fst i
      hy : LE.le (Quotient.out y).fst i
      xy : Eq ((g i) ((f (Quotient.out x).fst i hx) (Quotient.out x).snd)) ((g i) (( …
      ⊢ Eq ((f (Quotient.out x).fst i hx) (Quotient.out x).snd) ((f (Quotient.out y) …
    -/
    exact (g i).injective xy
    /-
      🎉 no goals
    -/
  map_fun' F x := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      F : L.Functions n✝
      x : Fin n✝ → FirstOrder.Language.DirectLimit G f
      ⊢ Eq ({ toFun := Quotient.lift (fun x => (g x.fst) x.snd) ⋯, inj' := ⋯ }.toFun …
    -/
    obtain ⟨i, y, rfl⟩ := exists_quotient_mk'_sigma_mk'_eq G f x
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      F : L.Functions n✝
      i : ι
      y : Fin n✝ → G i
      ⊢ Eq ({ toFun := Quotient.lift (fun x => (g x.fst) x.snd) ⋯, inj' := ⋯ }.toFun …
    -/
    change _ = funMap F (Quotient.lift _ _ ∘ Quotient.mk _ ∘ Structure.Sigma.mk f i ∘ y)
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      F : L.Functions n✝
      i : ι
      y : Fin n✝ → G i
      ⊢ Eq ({ toFun := Quotient.lift (fun x => (g x.fst) x.snd) ⋯, inj' := ⋯ }.toFun …
    -/
    rw [funMap_quotient_mk'_sigma_mk', ← Function.comp_assoc, Quotient.lift_comp_mk]
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      F : L.Functions n✝
      i : ι
      y : Fin n✝ → G i
      ⊢ Eq ({ toFun := Quotient.lift (fun x => (g x.fst) x.snd) ⋯, inj' := ⋯ }.toFun …
    -/
    simp only [Quotient.lift_mk, Embedding.map_fun]
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      F : L.Functions n✝
      i : ι
      y : Fin n✝ → G i
      ⊢ Eq (FirstOrder.Language.Structure.funMap F (Function.comp (⇑(g i)) y)) (Firs …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_rel' R x := by
    /-
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      R : L.Relations n✝
      x : Fin n✝ → FirstOrder.Language.DirectLimit G f
      ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp { toFun := Quotie …
    -/
    obtain ⟨i, y, rfl⟩ := exists_quotient_mk'_sigma_mk'_eq G f x
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      R : L.Relations n✝
      i : ι
      y : Fin n✝ → G i
      ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp { toFun := Quotie …
    -/
    change RelMap R (Quotient.lift _ _ ∘ Quotient.mk _ ∘ Structure.Sigma.mk f i ∘ y) ↔ _
    rw [relMap_quotient_mk'_sigma_mk' G f, ← (g i).map_rel R y, ← Function.comp_assoc,
      Quotient.lift_comp_mk]
    /-
      case intro.intro
      L : FirstOrder.Language
      ι : Type v
      inst✝⁵ : Preorder ι
      G : ι → Type w
      inst✝⁴ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝¹ : Nonempty ι
      P : Type u₁
      inst✝ : L.Structure P
      g : (i : ι) → L.Embedding (G i) P
      Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
      n✝ : Nat
      R : L.Relations n✝
      i : ι
      y : Fin n✝ → G i
      ⊢ Iff (FirstOrder.Language.Structure.RelMap R (Function.comp (fun x => (g x.fs …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_quotient_mk'_sigma_mk' {i} (x : G i) : lift L ι G f g Hg ⟦.mk f i x⟧ = (g i) x := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    P : Type u₁
    inst✝ : L.Structure P
    g : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    i : ι
    x : G i
    ⊢ Eq ((FirstOrder.Language.DirectLimit.lift L ι G f g Hg) (Quotient.mk (FirstO …
  -/
  change (lift L ι G f g Hg).toFun ⟦.mk f i x⟧ = _
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    P : Type u₁
    inst✝ : L.Structure P
    g : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    i : ι
    x : G i
    ⊢ Eq ((FirstOrder.Language.DirectLimit.lift L ι G f g Hg).toFun (Quotient.mk ( …
  -/
  simp only [lift, Quotient.lift_mk]
  /-
    🎉 no goals
  -/


                                                                                 /-
                                                                                   L : FirstOrder.Language
                                                                                   ι : Type v
                                                                                   inst✝⁵ : Preorder ι
                                                                                   G : ι → Type w
                                                                                   inst✝⁴ : (i : ι) → L.Structure (G i)
                                                                                   f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
                                                                                   inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                                                   inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
                                                                                   inst✝¹ : Nonempty ι
                                                                                   P : Type u₁
                                                                                   inst✝ : L.Structure P
                                                                                   g : (i : ι) → L.Embedding (G i) P
                                                                                   Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                                                                                   i : ι
                                                                                   x : G i
                                                                                   ⊢ Eq ((FirstOrder.Language.DirectLimit.lift L ι G f g Hg) ((FirstOrder.Languag …
                                                                                 -/
theorem lift_of {i} (x : G i) : lift L ι G f g Hg (of L ι G f i x) = g i x := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem lift_unique (F : DirectLimit G f ↪[L] P) (x) :
    F x =
      lift L ι G f (fun i => F.comp <| of L ι G f i)
                             /-
                               L : FirstOrder.Language
                               ι : Type v
                               inst✝⁵ : Preorder ι
                               G : ι → Type w
                               inst✝⁴ : (i : ι) → L.Structure (G i)
                               f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
                               inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                               inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
                               inst✝¹ : Nonempty ι
                               P : Type u₁
                               inst✝ : L.Structure P
                               g : (i : ι) → L.Embedding (G i) P
                               Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
                               F : L.Embedding (FirstOrder.Language.DirectLimit G f) P
                               x✝ : FirstOrder.Language.DirectLimit G f
                               i j : ι
                               hij : LE.le i j
                               x : G i
                               ⊢ Eq (((fun i => F.comp (FirstOrder.Language.DirectLimit.of L ι G f i)) j) ((f …
                             -/
        (fun i j hij x => by rw [F.comp_apply, F.comp_apply, of_f]) x :=
                             /-
                               🎉 no goals
                             -/
                                          /-
                                            L : FirstOrder.Language
                                            ι : Type v
                                            inst✝⁵ : Preorder ι
                                            G : ι → Type w
                                            inst✝⁴ : (i : ι) → L.Structure (G i)
                                            f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
                                            inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                            inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
                                            inst✝¹ : Nonempty ι
                                            P : Type u₁
                                            inst✝ : L.Structure P
                                            F : L.Embedding (FirstOrder.Language.DirectLimit G f) P
                                            x✝ : FirstOrder.Language.DirectLimit G f
                                            i : ι
                                            x : G i
                                            ⊢ Eq (F ((FirstOrder.Language.DirectLimit.of L ι G f i) x)) ((FirstOrder.Langu …
                                          -/
  DirectLimit.inductionOn x fun i x => by rw [lift_of]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma range_lift : (lift L ι G f g Hg).toHom.range = ⨆ i, (g i).toHom.range := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    P : Type u₁
    inst✝ : L.Structure P
    g : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    ⊢ Eq (FirstOrder.Language.DirectLimit.lift L ι G f g Hg).toHom.range (iSup fun …
  -/
  simp_rw [Hom.range_eq_map]
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    P : Type u₁
    inst✝ : L.Structure P
    g : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    ⊢ Eq (FirstOrder.Language.Substructure.map (FirstOrder.Language.DirectLimit.li …
  -/
  rw [← iSup_range_of_eq_top, Substructure.map_iSup]
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    P : Type u₁
    inst✝ : L.Structure P
    g : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    ⊢ Eq (iSup fun i => FirstOrder.Language.Substructure.map (FirstOrder.Language. …
  -/
  simp_rw [Hom.range_eq_map, Substructure.map_map]
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁵ : Preorder ι
    G : ι → Type w
    inst✝⁴ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝¹ : Nonempty ι
    P : Type u₁
    inst✝ : L.Structure P
    g : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij) x)) ((g i …
    ⊢ Eq (iSup fun i => FirstOrder.Language.Substructure.map ((FirstOrder.Language …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The isomorphism between limits of isomorphic systems. -/
noncomputable def equiv_lift (H_commuting : ∀ i j hij x, g j (f i j hij x) = f' i j hij (g i x)) :
    DirectLimit G f ≃[L] DirectLimit G' f' := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁷ : Preorder ι
    G : ι → Type w
    inst✝⁶ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝⁵ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝⁴ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝³ : Nonempty ι
    P : Type u₁
    inst✝² : L.Structure P
    g✝ : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝ j) ((f i j hij) x)) ((g✝ …
    G' : ι → Type w'
    inst✝¹ : (i : ι) → L.Structure (G' i)
    f' : (i j : ι) → LE.le i j → L.Embedding (G' i) (G' j)
    g : (i : ι) → L.Equiv (G i) (G' i)
    inst✝ : DirectedSystem G' fun i j h => ⇑(f' i j h)
    H_commuting : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij)  …
    ⊢ L.Equiv (FirstOrder.Language.DirectLimit G f) (FirstOrder.Language.DirectLim …
  -/
  let U i : G i ↪[L] DirectLimit G' f' := (of L _ G' f' i).comp (g i).toEmbedding
  let F : DirectLimit G f ↪[L] DirectLimit G' f' := lift L _ G f U <| by
    intro _ _ _ _
    simp only [U, Embedding.comp_apply, Equiv.coe_toEmbedding, H_commuting, of_f]
  have surj_f : Function.Surjective F := by
    intro x
    rcases x with ⟨i, pre_x⟩
    use of L _ G f i ((g i).symm pre_x)
    simp only [F, U, lift_of, Embedding.comp_apply, Equiv.coe_toEmbedding, Equiv.apply_symm_apply]
    rfl
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁷ : Preorder ι
    G : ι → Type w
    inst✝⁶ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝⁵ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝⁴ : DirectedSystem G fun i j h => ⇑(f i j h)
    inst✝³ : Nonempty ι
    P : Type u₁
    inst✝² : L.Structure P
    g✝ : (i : ι) → L.Embedding (G i) P
    Hg : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g✝ j) ((f i j hij) x)) ((g✝ …
    G' : ι → Type w'
    inst✝¹ : (i : ι) → L.Structure (G' i)
    f' : (i j : ι) → LE.le i j → L.Embedding (G' i) (G' j)
    g : (i : ι) → L.Equiv (G i) (G' i)
    inst✝ : DirectedSystem G' fun i j h => ⇑(f' i j h)
    H_commuting : ∀ (i j : ι) (hij : LE.le i j) (x : G i), Eq ((g j) ((f i j hij)  …
    U : (i : ι) → L.Embedding (G i) (FirstOrder.Language.DirectLimit G' f') := fun …
    F : L.Embedding (FirstOrder.Language.DirectLimit G f) (FirstOrder.Language.Dir …
    surj_f : Function.Surjective ⇑F
    ⊢ L.Equiv (FirstOrder.Language.DirectLimit G f) (FirstOrder.Language.DirectLim …
  -/
  exact ⟨Equiv.ofBijective F ⟨F.injective, surj_f⟩, F.map_fun', F.map_rel'⟩
  /-
    🎉 no goals
  -/


theorem equiv_lift_of {i : ι} (x : G i) :
    equiv_lift L ι G f G' f' g H_commuting (of L ι G f i x) = of L ι G' f' i (g i x) := rfl


/-- The direct limit of countably many countably generated structures is countably generated. -/
theorem cg {ι : Type*} [Countable ι] [Preorder ι] [IsDirected ι (· ≤ ·)] [Nonempty ι]
    {G : ι → Type w} [∀ i, L.Structure (G i)] (f : ∀ i j, i ≤ j → G i ↪[L] G j)
    (h : ∀ i, Structure.CG L (G i)) [DirectedSystem G fun i j h => f i j h] :
    Structure.CG L (DirectLimit G f) := by
  /-
    L : FirstOrder.Language
    ι : Type u_1
    inst✝⁵ : Countable ι
    inst✝⁴ : Preorder ι
    inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝² : Nonempty ι
    G : ι → Type w
    inst✝¹ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
    inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
    ⊢ FirstOrder.Language.Structure.CG L (FirstOrder.Language.DirectLimit G f)
  -/
  refine ⟨⟨⋃ i, DirectLimit.of L ι G f i '' Classical.choose (h i).out, ?_, ?_⟩⟩
    /-
      case refine_1
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      ⊢ (Set.iUnion fun i => Set.image (⇑(FirstOrder.Language.DirectLimit.of L ι G f …
    -/
  · exact Set.countable_iUnion fun i => Set.Countable.image (Classical.choose_spec (h i).out).1 _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      ⊢ Eq ((FirstOrder.Language.Substructure.closure L).toFun (Set.iUnion fun i =>  …
    -/
  · rw [eq_top_iff, Substructure.closure_iUnion]
    /-
      case refine_2
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      ⊢ LE.le Top.top (iSup fun i => (FirstOrder.Language.Substructure.closure L).to …
    -/
    simp_rw [← Embedding.coe_toHom, Substructure.closure_image]
    /-
      case refine_2
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      ⊢ LE.le Top.top (iSup fun i => FirstOrder.Language.Substructure.map (FirstOrde …
    -/
    rw [le_iSup_iff]
    /-
      case refine_2
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      ⊢ ∀ (b : L.Substructure (FirstOrder.Language.DirectLimit G f)), (∀ (i : ι), LE …
    -/
    intro S hS x _
    /-
      case refine_2
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      S : L.Substructure (FirstOrder.Language.DirectLimit G f)
      hS : ∀ (i : ι), LE.le (FirstOrder.Language.Substructure.map (FirstOrder.Langua …
      x : FirstOrder.Language.DirectLimit G f
      a✝ : Membership.mem Top.top x
      ⊢ Membership.mem S x
    -/
    let out := Quotient.out (s := DirectLimit.setoid G f)
    /-
      case refine_2
      L : FirstOrder.Language
      ι : Type u_1
      inst✝⁵ : Countable ι
      inst✝⁴ : Preorder ι
      inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
      inst✝² : Nonempty ι
      G : ι → Type w
      inst✝¹ : (i : ι) → L.Structure (G i)
      f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
      h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      S : L.Substructure (FirstOrder.Language.DirectLimit G f)
      hS : ∀ (i : ι), LE.le (FirstOrder.Language.Substructure.map (FirstOrder.Langua …
      x : FirstOrder.Language.DirectLimit G f
      a✝ : Membership.mem Top.top x
      out : Quotient (FirstOrder.Language.DirectLimit.setoid G f) → FirstOrder.Langu …
      ⊢ Membership.mem S x
    -/
    refine hS (out x).1 ⟨(out x).2, ?_, ?_⟩
      /-
        case refine_2.refine_1
        L : FirstOrder.Language
        ι : Type u_1
        inst✝⁵ : Countable ι
        inst✝⁴ : Preorder ι
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : Nonempty ι
        G : ι → Type w
        inst✝¹ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
        inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
        S : L.Substructure (FirstOrder.Language.DirectLimit G f)
        hS : ∀ (i : ι), LE.le (FirstOrder.Language.Substructure.map (FirstOrder.Langua …
        x : FirstOrder.Language.DirectLimit G f
        a✝ : Membership.mem Top.top x
        out : Quotient (FirstOrder.Language.DirectLimit.setoid G f) → FirstOrder.Langu …
        ⊢ Membership.mem (↑((FirstOrder.Language.Substructure.closure L).toFun (Classi …
      -/
    · rw [(Classical.choose_spec (h (out x).1).out).2]
      /-
        case refine_2.refine_1
        L : FirstOrder.Language
        ι : Type u_1
        inst✝⁵ : Countable ι
        inst✝⁴ : Preorder ι
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : Nonempty ι
        G : ι → Type w
        inst✝¹ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
        inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
        S : L.Substructure (FirstOrder.Language.DirectLimit G f)
        hS : ∀ (i : ι), LE.le (FirstOrder.Language.Substructure.map (FirstOrder.Langua …
        x : FirstOrder.Language.DirectLimit G f
        a✝ : Membership.mem Top.top x
        out : Quotient (FirstOrder.Language.DirectLimit.setoid G f) → FirstOrder.Langu …
        ⊢ Membership.mem (↑Top.top) (out x).snd
      -/
      trivial
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        L : FirstOrder.Language
        ι : Type u_1
        inst✝⁵ : Countable ι
        inst✝⁴ : Preorder ι
        inst✝³ : IsDirected ι fun x1 x2 => LE.le x1 x2
        inst✝² : Nonempty ι
        G : ι → Type w
        inst✝¹ : (i : ι) → L.Structure (G i)
        f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
        h : ∀ (i : ι), FirstOrder.Language.Structure.CG L (G i)
        inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
        S : L.Substructure (FirstOrder.Language.DirectLimit G f)
        hS : ∀ (i : ι), LE.le (FirstOrder.Language.Substructure.map (FirstOrder.Langua …
        x : FirstOrder.Language.DirectLimit G f
        a✝ : Membership.mem Top.top x
        out : Quotient (FirstOrder.Language.DirectLimit.setoid G f) → FirstOrder.Langu …
        ⊢ Eq ((FirstOrder.Language.DirectLimit.of L ι G f (out x).fst).toHom (out x).s …
      -/
    · simp only [out, Embedding.coe_toHom, DirectLimit.of_apply, Sigma.eta, Quotient.out_eq]
      /-
        🎉 no goals
      -/


instance cg' {ι : Type*} [Countable ι] [Preorder ι] [IsDirected ι (· ≤ ·)] [Nonempty ι]
    {G : ι → Type w} [∀ i, L.Structure (G i)] (f : ∀ i j, i ≤ j → G i ↪[L] G j)
    [h : ∀ i, Structure.CG L (G i)] [DirectedSystem G fun i j h => f i j h] :
    Structure.CG L (DirectLimit G f) :=
  cg f h


instance : DirectedSystem (fun i ↦ S i) (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) where
  map_self _ _ := rfl
  map_map _ _ _ _ _ _ := rfl


/-- The map from a direct limit of a system of substructures of `M` into `M`. -/
def liftInclusion :
    DirectLimit (fun i ↦ S i) (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) ↪[L] M :=
  DirectLimit.lift L ι (fun i ↦ S i) (fun _ _ h ↦ Substructure.inclusion (S.monotone h))
    (fun _ ↦ Substructure.subtype _) (fun _ _ _ _ ↦ rfl)


theorem liftInclusion_of {i : ι} (x : S i) :
    (liftInclusion S) (of L ι _ (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) i x)
    = Substructure.subtype (S i) x := rfl


lemma rangeLiftInclusion : (liftInclusion S).toHom.range = ⨆ i, S i := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    ⊢ Eq (FirstOrder.Language.DirectLimit.liftInclusion S).toHom.range (iSup fun i …
  -/
  simp_rw [liftInclusion, range_lift, Substructure.range_subtype]
  /-
    🎉 no goals
  -/


/-- The isomorphism between a direct limit of a system of substructures and their union. -/
noncomputable def Equiv_iSup :
    DirectLimit (fun i ↦ S i) (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) ≃[L]
    (iSup S : L.Substructure M) := by
  have liftInclusion_in_sup : ∀ x, liftInclusion S x ∈ (⨆ i, S i) := by
    simp only [← rangeLiftInclusion, Hom.mem_range, Embedding.coe_toHom]
    intro x; use x
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    liftInclusion_in_sup : ∀ (x : FirstOrder.Language.DirectLimit (fun i => Subtyp …
    ⊢ L.Equiv (FirstOrder.Language.DirectLimit (fun i => Subtype fun x => Membersh …
  -/
  let F := Embedding.codRestrict (⨆ i, S i) _ liftInclusion_in_sup
  have F_surj : Function.Surjective F := by
    rintro ⟨m, hm⟩
    rw [← rangeLiftInclusion, Hom.mem_range] at hm
    rcases hm with ⟨a, _⟩; use a
    simpa only [F, Embedding.codRestrict_apply', Subtype.mk.injEq]
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝⁴ : Preorder ι
    G : ι → Type w
    inst✝³ : (i : ι) → L.Structure (G i)
    f : (i j : ι) → LE.le i j → L.Embedding (G i) (G j)
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    liftInclusion_in_sup : ∀ (x : FirstOrder.Language.DirectLimit (fun i => Subtyp …
    F : L.Embedding (FirstOrder.Language.DirectLimit (fun i => Subtype fun x => Me …
    F_surj : Function.Surjective ⇑F
    ⊢ L.Equiv (FirstOrder.Language.DirectLimit (fun i => Subtype fun x => Membersh …
  -/
  exact ⟨Equiv.ofBijective F ⟨F.injective, F_surj⟩, F.map_fun', F.map_rel'⟩
  /-
    🎉 no goals
  -/


theorem Equiv_isup_of_apply {i : ι} (x : S i) :
    Equiv_iSup S (of L ι _ (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) i x)
    = Substructure.inclusion (le_iSup _ _) x := rfl


theorem Equiv_isup_symm_inclusion_apply {i : ι} (x : S i) :
    (Equiv_iSup S).symm (Substructure.inclusion (le_iSup _ _) x)
    = of L ι _ (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) i x := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    i : ι
    x : Subtype fun x => Membership.mem (S i) x
    ⊢ Eq ((FirstOrder.Language.DirectLimit.Equiv_iSup S).symm ((FirstOrder.Languag …
  -/
  apply (Equiv_iSup S).injective
  /-
    case a
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    i : ι
    x : Subtype fun x => Membership.mem (S i) x
    ⊢ Eq ((FirstOrder.Language.DirectLimit.Equiv_iSup S) ((FirstOrder.Language.Dir …
  -/
  simp only [Equiv.apply_symm_apply]
  /-
    case a
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    i : ι
    x : Subtype fun x => Membership.mem (S i) x
    ⊢ Eq ((FirstOrder.Language.Substructure.inclusion ⋯) x) ((FirstOrder.Language. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem Equiv_isup_symm_inclusion (i : ι) :
    (Equiv_iSup S).symm.toEmbedding.comp (Substructure.inclusion (le_iSup _ _))
    = of L ι _ (fun _ _ h ↦ Substructure.inclusion (S.monotone h)) i := by
  /-
    L : FirstOrder.Language
    ι : Type v
    inst✝³ : Preorder ι
    inst✝² : Nonempty ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    M : Type u_1
    inst✝ : L.Structure M
    S : OrderHom ι (L.Substructure M)
    i : ι
    ⊢ Eq ((FirstOrder.Language.DirectLimit.Equiv_iSup S).symm.toEmbedding.comp (Fi …
  -/
  ext x; exact Equiv_isup_symm_inclusion_apply _ x
         /-
           🎉 no goals
         -/


