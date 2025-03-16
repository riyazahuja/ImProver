/-- Given `α : ι → Sort*`, `Pi.nil α` is the trivial dependent function out of the empty list. -/
def nil (α : ι → Sort*) : (∀ i ∈ ([] : List ι), α i) :=
  nofun


/-- Given `f` a function whose domain is `i :: l`, get its value at `i`. -/
def head (f : ∀ j ∈ i :: l, α j) : α i :=
  f i (mem_cons_self _ _)


/-- Given `f` a function whose domain is `i :: l`, produce a function whose domain
is restricted to `l`. -/
def tail (f : ∀ j ∈ i :: l, α j) : ∀ j ∈ l, α j :=
  fun j hj ↦ f j (mem_cons_of_mem _ hj)


/-- Given `α : ι → Sort*`, a list `l` and a term `i`, as well as a term `a : α i` and a
function `f` such that `f j : α j` for all `j` in `l`, `Pi.cons a f` is a function `g` such
that `g k : α k` for all `k` in `i :: l`. -/
def cons (a : α i) (f : ∀ j ∈ l, α j) : ∀ j ∈ i :: l, α j :=
  Multiset.Pi.cons (α := ι) l _ a f


lemma cons_def (a : α i) (f : ∀ j ∈ l, α j) : cons _ _ a f =
    fun j hj ↦ if h : j = i then h.symm.rec a else f j <| (mem_cons.1 hj).resolve_left h :=
  rfl


@[simp] lemma _root_.Multiset.Pi.cons_coe {l : List ι} (a : α i) (f : ∀ j ∈ l, α j) :
    Multiset.Pi.cons l _ a f = cons _ _ a f :=
  rfl


@[simp] lemma cons_eta (f : ∀ j ∈ i :: l, α j) :
    cons _ _ (head f) (tail f) = f :=
  Multiset.Pi.cons_eta (α := ι) (m := l) f


lemma cons_map (a : α i) (f : ∀ j ∈ l, α j)
    {α' : ι → Sort*} (φ : ∀ ⦃j⦄, α j → α' j) :
    cons _ _ (φ a) (fun j hj ↦ φ (f j hj)) = (fun j hj ↦ φ ((cons _ _ a f) j hj)) :=
  Multiset.Pi.cons_map _ _ _


lemma forall_rel_cons_ext {r : ∀ ⦃i⦄, α i → α i → Prop} {a₁ a₂ : α i} {f₁ f₂ : ∀ j ∈ l, α j}
    (ha : r a₁ a₂) (hf : ∀ (i : ι) (hi : i ∈ l), r (f₁ i hi) (f₂ i hi)) :
    ∀ j hj, r (cons _ _ a₁ f₁ j hj) (cons _ _ a₂ f₂ j hj) :=
  Multiset.Pi.forall_rel_cons_ext (α := ι) (m := l) ha hf


/-- `pi xs f` creates the list of functions `g` such that, for `x ∈ xs`, `g x ∈ f x` -/
def pi : ∀ l : List ι, (∀ i, List (α i)) → List (∀ i, i ∈ l → α i)
  |     [],  _ => [List.Pi.nil α]
  | i :: l, fs => (fs i).flatMap (fun b ↦ (pi l fs).map (List.Pi.cons _ _ b))


@[simp] lemma pi_nil (t : ∀ i, List (α i)) :
    pi [] t = [Pi.nil α] :=
  rfl


@[simp] lemma pi_cons (i : ι) (l : List ι) (t : ∀ j, List (α j)) :
    pi (i :: l) t = ((t i).flatMap fun b ↦ (pi l t).map <| Pi.cons _ _ b) :=
  rfl


lemma _root_.Multiset.pi_coe (l : List ι) (fs : ∀ i, List (α i)) :
    (l : Multiset ι).pi (fs ·) = (↑(pi l fs) : Multiset (∀ i ∈ l, α i)) := by
  /-
    ι : Type u_1
    inst✝ : DecidableEq ι
    α : ι → Type u_2
    l : List ι
    fs : (i : ι) → List (α i)
    ⊢ Eq ((↑l).pi fun x => ↑(fs x)) ↑(l.pi fs)
  -/
  induction' l with i l ih
    /-
      case nil
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Type u_2
      fs : (i : ι) → List (α i)
      ⊢ Eq ((↑List.nil).pi fun x => ↑(fs x)) ↑(List.nil.pi fs)
    -/
  · change Multiset.pi 0 _ = _
    /-
      case nil
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Type u_2
      fs : (i : ι) → List (α i)
      ⊢ Eq (Multiset.pi 0 fun x => ↑(fs x)) ↑(List.nil.pi fs)
    -/
    simp only [Multiset.coe_singleton, Multiset.pi_zero, pi_nil, Multiset.singleton_inj]
    /-
      case nil
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Type u_2
      fs : (i : ι) → List (α i)
      ⊢ Eq (Multiset.Pi.empty α) (List.Pi.nil α)
    -/
    ext i hi
    /-
      case nil.h.h
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Type u_2
      fs : (i : ι) → List (α i)
      i : ι
      hi : Membership.mem 0 i
      ⊢ Eq (Multiset.Pi.empty α i hi) (List.Pi.nil α i hi)
    -/
    simp at hi
    /-
      🎉 no goals
    -/
    /-
      case cons
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Type u_2
      fs : (i : ι) → List (α i)
      i : ι
      l : List ι
      ih : Eq ((↑l).pi fun x => ↑(fs x)) ↑(l.pi fs)
      ⊢ Eq ((↑(List.cons i l)).pi fun x => ↑(fs x)) ↑((List.cons i l).pi fs)
    -/
  · change Multiset.pi (i ::ₘ ↑l) _ = _
    /-
      case cons
      ι : Type u_1
      inst✝ : DecidableEq ι
      α : ι → Type u_2
      fs : (i : ι) → List (α i)
      i : ι
      l : List ι
      ih : Eq ((↑l).pi fun x => ↑(fs x)) ↑(l.pi fs)
      ⊢ Eq ((Multiset.cons i ↑l).pi fun x => ↑(fs x)) ↑((List.cons i l).pi fs)
    -/
    simp [ih, Multiset.coe_bind, - Multiset.cons_coe]
    /-
      🎉 no goals
    -/


lemma mem_pi {l : List ι} (fs : ∀ i, List (α i)) :
    ∀ f : ∀ i ∈ l, α i, (f ∈ pi l fs) ↔ (∀ i (hi : i ∈ l), f i hi ∈ fs i) := by
  /-
    ι : Type u_1
    inst✝ : DecidableEq ι
    α : ι → Type u_2
    l : List ι
    fs : (i : ι) → List (α i)
    ⊢ ∀ (f : (i : ι) → Membership.mem l i → α i), Iff (Membership.mem (l.pi fs) f) …
  -/
  intros f
  /-
    ι : Type u_1
    inst✝ : DecidableEq ι
    α : ι → Type u_2
    l : List ι
    fs : (i : ι) → List (α i)
    f : (i : ι) → Membership.mem l i → α i
    ⊢ Iff (Membership.mem (l.pi fs) f) (∀ (i : ι) (hi : Membership.mem l i), Membe …
  -/
  convert @Multiset.mem_pi ι _ α ↑l (fs ·) f using 1
  /-
    case h.e'_1.a
    ι : Type u_1
    inst✝ : DecidableEq ι
    α : ι → Type u_2
    l : List ι
    fs : (i : ι) → List (α i)
    f : (i : ι) → Membership.mem l i → α i
    ⊢ Iff (Membership.mem (l.pi fs) f) (Membership.mem ((↑l).pi fun x => ↑(fs x)) f)
  -/
  rw [Multiset.pi_coe]
  /-
    case h.e'_1.a
    ι : Type u_1
    inst✝ : DecidableEq ι
    α : ι → Type u_2
    l : List ι
    fs : (i : ι) → List (α i)
    f : (i : ι) → Membership.mem l i → α i
    ⊢ Iff (Membership.mem (l.pi fs) f) (Membership.mem (↑(l.pi fs)) f)
  -/
  rfl
  /-
    🎉 no goals
  -/


