/-- The product of a family of types over a list. -/
abbrev TProd (l : List ι) : Type v :=
  l.foldr (fun i β => α i × β) PUnit


/-- Turning a function `f : ∀ i, α i` into an element of the iterated product `TProd α l`. -/
protected def mk : ∀ (l : List ι) (_f : ∀ i, α i), TProd α l
  | [] => fun _ => PUnit.unit
  | i :: is => fun f => (f i, TProd.mk is f)


instance [∀ i, Inhabited (α i)] : Inhabited (TProd α l) :=
  ⟨TProd.mk l default⟩


@[simp]
theorem fst_mk (i : ι) (l : List ι) (f : ∀ i, α i) : (TProd.mk (i :: l) f).1 = f i :=
  rfl


@[simp]
theorem snd_mk (i : ι) (l : List ι) (f : ∀ i, α i) :
    (TProd.mk.{u,v} (i :: l) f).2 = TProd.mk.{u,v} l f :=
  rfl


/-- Given an element of the iterated product `l.Prod α`, take a projection into direction `i`.
  If `i` appears multiple times in `l`, this chooses the first component in direction `i`. -/
protected def elim : ∀ {l : List ι} (_ : TProd α l) {i : ι} (_ : i ∈ l), α i
  | i :: is, v, j, hj =>
    if hji : j = i then by
      /-
        ι : Type u
        α : ι → Type v
        i✝ j✝ : ι
        l : List ι
        inst✝ : DecidableEq ι
        i : ι
        is : List ι
        v : List.TProd α (List.cons i is)
        j : ι
        hj : Membership.mem (List.cons i is) j
        hji : Eq j i
        ⊢ α j
      -/
      subst hji
      /-
        ι : Type u
        α : ι → Type v
        i j✝ : ι
        l : List ι
        inst✝ : DecidableEq ι
        is : List ι
        j : ι
        v : List.TProd α (List.cons j is)
        hj : Membership.mem (List.cons j is) j
        ⊢ α j
      -/
      exact v.1
      /-
        🎉 no goals
      -/
    else TProd.elim v.2 ((List.mem_cons.mp hj).resolve_left hji)


@[simp]
                                                                                  /-
                                                                                    ι : Type u
                                                                                    α : ι → Type v
                                                                                    i : ι
                                                                                    l : List ι
                                                                                    inst✝ : DecidableEq ι
                                                                                    v : List.TProd α (List.cons i l)
                                                                                    ⊢ Eq (v.elim ⋯) v.1
                                                                                  -/
theorem elim_self (v : TProd α (i :: l)) : v.elim (l.mem_cons_self i) = v.1 := by simp [TProd.elim]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem elim_of_ne (hj : j ∈ i :: l) (hji : j ≠ i) (v : TProd α (i :: l)) :
                                                                              /-
                                                                                ι : Type u
                                                                                α : ι → Type v
                                                                                i j : ι
                                                                                l : List ι
                                                                                inst✝ : DecidableEq ι
                                                                                hj : Membership.mem (List.cons i l) j
                                                                                hji : Ne j i
                                                                                v : List.TProd α (List.cons i l)
                                                                                ⊢ Eq (v.elim hj) (List.TProd.elim v.2 ⋯)
                                                                              -/
    v.elim hj = TProd.elim v.2 ((List.mem_cons.mp hj).resolve_left hji) := by simp [TProd.elim, hji]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem elim_of_mem (hl : (i :: l).Nodup) (hj : j ∈ l) (v : TProd α (i :: l)) :
    v.elim (mem_cons_of_mem _ hj) = TProd.elim v.2 hj := by
  /-
    ι : Type u
    α : ι → Type v
    i j : ι
    l : List ι
    inst✝ : DecidableEq ι
    hl : (List.cons i l).Nodup
    hj : Membership.mem l j
    v : List.TProd α (List.cons i l)
    ⊢ Eq (v.elim ⋯) (List.TProd.elim v.2 hj)
  -/
  apply elim_of_ne
  /-
    case hji
    ι : Type u
    α : ι → Type v
    i j : ι
    l : List ι
    inst✝ : DecidableEq ι
    hl : (List.cons i l).Nodup
    hj : Membership.mem l j
    v : List.TProd α (List.cons i l)
    ⊢ Ne j i
  -/
  rintro rfl
  /-
    case hji
    ι : Type u
    α : ι → Type v
    j : ι
    l : List ι
    inst✝ : DecidableEq ι
    hj : Membership.mem l j
    hl : (List.cons j l).Nodup
    v : List.TProd α (List.cons j l)
    ⊢ False
  -/
  exact hl.not_mem hj
  /-
    🎉 no goals
  -/


theorem elim_mk : ∀ (l : List ι) (f : ∀ i, α i) {i : ι} (hi : i ∈ l), (TProd.mk l f).elim hi = f i
  | i :: is, f, j, hj => by
    /-
      ι : Type u
      α : ι → Type v
      inst✝ : DecidableEq ι
      i : ι
      is : List ι
      f : (i : ι) → α i
      j : ι
      hj : Membership.mem (List.cons i is) j
      ⊢ Eq ((List.TProd.mk (List.cons i is) f).elim hj) (f j)
    -/
    by_cases hji : j = i
      /-
        case pos
        ι : Type u
        α : ι → Type v
        inst✝ : DecidableEq ι
        i : ι
        is : List ι
        f : (i : ι) → α i
        j : ι
        hj : Membership.mem (List.cons i is) j
        hji : Eq j i
        ⊢ Eq ((List.TProd.mk (List.cons i is) f).elim hj) (f j)
      -/
    · subst hji
      /-
        case pos
        ι : Type u
        α : ι → Type v
        inst✝ : DecidableEq ι
        is : List ι
        f : (i : ι) → α i
        j : ι
        hj : Membership.mem (List.cons j is) j
        ⊢ Eq ((List.TProd.mk (List.cons j is) f).elim hj) (f j)
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        α : ι → Type v
        inst✝ : DecidableEq ι
        i : ι
        is : List ι
        f : (i : ι) → α i
        j : ι
        hj : Membership.mem (List.cons i is) j
        hji : Not (Eq j i)
        ⊢ Eq ((List.TProd.mk (List.cons i is) f).elim hj) (f j)
      -/
    · rw [TProd.elim_of_ne _ hji, snd_mk, elim_mk is]
      /-
        🎉 no goals
      -/


@[ext]
theorem ext :
    ∀ {l : List ι} (_ : l.Nodup) {v w : TProd α l}
      (_ : ∀ (i) (hi : i ∈ l), v.elim hi = w.elim hi), v = w
  | [], _, v, w, _ => PUnit.ext v w
  | i :: is, hl, v, w, hvw => by
    /-
      ι : Type u
      α : ι → Type v
      inst✝ : DecidableEq ι
      i : ι
      is : List ι
      hl : (List.cons i is).Nodup
      v w : List.TProd α (List.cons i is)
      hvw : ∀ (i_1 : ι) (hi : Membership.mem (List.cons i is) i_1), Eq (v.elim hi) ( …
      ⊢ Eq v w
    -/
    apply Prod.ext
      /-
        case fst
        ι : Type u
        α : ι → Type v
        inst✝ : DecidableEq ι
        i : ι
        is : List ι
        hl : (List.cons i is).Nodup
        v w : List.TProd α (List.cons i is)
        hvw : ∀ (i_1 : ι) (hi : Membership.mem (List.cons i is) i_1), Eq (v.elim hi) ( …
        ⊢ Eq v.1 w.1
      -/
    · rw [← elim_self v, hvw, elim_self]
      /-
        🎉 no goals
      -/
    /-
      case snd
      ι : Type u
      α : ι → Type v
      inst✝ : DecidableEq ι
      i : ι
      is : List ι
      hl : (List.cons i is).Nodup
      v w : List.TProd α (List.cons i is)
      hvw : ∀ (i_1 : ι) (hi : Membership.mem (List.cons i is) i_1), Eq (v.elim hi) ( …
      ⊢ Eq v.2 w.2
    -/
    refine ext (nodup_cons.mp hl).2 fun j hj => ?_
    /-
      case snd
      ι : Type u
      α : ι → Type v
      inst✝ : DecidableEq ι
      i : ι
      is : List ι
      hl : (List.cons i is).Nodup
      v w : List.TProd α (List.cons i is)
      hvw : ∀ (i_1 : ι) (hi : Membership.mem (List.cons i is) i_1), Eq (v.elim hi) ( …
      j : ι
      hj : Membership.mem is j
      ⊢ Eq (List.TProd.elim v.2 hj) (List.TProd.elim w.2 hj)
    -/
    rw [← elim_of_mem hl, hvw, elim_of_mem hl]
    /-
      🎉 no goals
    -/


/-- A version of `TProd.elim` when `l` contains all elements. In this case we get a function into
  `Π i, α i`. -/
@[simp]
protected def elim' (h : ∀ i, i ∈ l) (v : TProd α l) (i : ι) : α i :=
  v.elim (h i)


theorem mk_elim (hnd : l.Nodup) (h : ∀ i, i ∈ l) (v : TProd α l) : TProd.mk l (v.elim' h) = v :=
                               /-
                                 ι : Type u
                                 α : ι → Type v
                                 l : List ι
                                 inst✝ : DecidableEq ι
                                 hnd : l.Nodup
                                 h : ∀ (i : ι), Membership.mem l i
                                 v : List.TProd α l
                                 i : ι
                                 hi : Membership.mem l i
                                 ⊢ Eq ((List.TProd.mk l (List.TProd.elim' h v)).elim hi) (v.elim hi)
                               -/
  TProd.ext hnd fun i hi => by simp [elim_mk]
                               /-
                                 🎉 no goals
                               -/


/-- Pi-types are equivalent to iterated products. -/
def piEquivTProd (hnd : l.Nodup) (h : ∀ i, i ∈ l) : (∀ i, α i) ≃ TProd α l :=
  ⟨TProd.mk l, TProd.elim' h, fun f => funext fun i => elim_mk l f (h i), mk_elim hnd h⟩


/-- A product of sets in `TProd α l`. -/
@[simp]
protected def tprod : ∀ (l : List ι) (_t : ∀ i, Set (α i)), Set (TProd α l)
  | [], _ => univ
  | i :: is, t => t i ×ˢ Set.tprod is t


theorem mk_preimage_tprod :
    ∀ (l : List ι) (t : ∀ i, Set (α i)), TProd.mk l ⁻¹' Set.tprod l t = { i | i ∈ l }.pi t
                /-
                  ι : Type u
                  α : ι → Type v
                  t : (i : ι) → Set (α i)
                  ⊢ Eq (Set.preimage (List.TProd.mk List.nil) (Set.tprod List.nil t)) ((setOf fu …
                -/
  | [], t => by simp [Set.tprod]
                /-
                  🎉 no goals
                -/
  | i :: l, t => by
    /-
      ι : Type u
      α : ι → Type v
      i : ι
      l : List ι
      t : (i : ι) → Set (α i)
      ⊢ Eq (Set.preimage (List.TProd.mk (List.cons i l)) (Set.tprod (List.cons i l)  …
    -/
    ext f
    have h : TProd.mk l f ∈ Set.tprod l t ↔ ∀ i : ι, i ∈ l → f i ∈ t i := by
      change f ∈ TProd.mk l ⁻¹' Set.tprod l t ↔ f ∈ { x | x ∈ l }.pi t
      rw [mk_preimage_tprod l t]

    -- `simp [Set.TProd, TProd.mk, this]` can close this goal but is slow.
    /-
      case h
      ι : Type u
      α : ι → Type v
      i : ι
      l : List ι
      t : (i : ι) → Set (α i)
      f : (i : ι) → α i
      h : Iff (Membership.mem (Set.tprod l t) (List.TProd.mk l f)) (∀ (i : ι), Membe …
      ⊢ Iff (Membership.mem (Set.preimage (List.TProd.mk (List.cons i l)) (Set.tprod …
    -/
    rw [Set.tprod, TProd.mk, mem_preimage, mem_pi, prod_mk_mem_set_prod_eq]
    /-
      case h
      ι : Type u
      α : ι → Type v
      i : ι
      l : List ι
      t : (i : ι) → Set (α i)
      f : (i : ι) → α i
      h : Iff (Membership.mem (Set.tprod l t) (List.TProd.mk l f)) (∀ (i : ι), Membe …
      ⊢ Iff (And (Membership.mem (t i) (f i)) (Membership.mem (Set.tprod l t) (List. …
    -/
    simp_rw [mem_setOf_eq, mem_cons]
    /-
      case h
      ι : Type u
      α : ι → Type v
      i : ι
      l : List ι
      t : (i : ι) → Set (α i)
      f : (i : ι) → α i
      h : Iff (Membership.mem (Set.tprod l t) (List.TProd.mk l f)) (∀ (i : ι), Membe …
      ⊢ Iff (And (Membership.mem (t i) (f i)) (Membership.mem (Set.tprod l t) (List. …
    -/
    rw [forall_eq_or_imp, and_congr_right_iff]
    /-
      case h
      ι : Type u
      α : ι → Type v
      i : ι
      l : List ι
      t : (i : ι) → Set (α i)
      f : (i : ι) → α i
      h : Iff (Membership.mem (Set.tprod l t) (List.TProd.mk l f)) (∀ (i : ι), Membe …
      ⊢ Membership.mem (t i) (f i) → Iff (Membership.mem (Set.tprod l t) (List.TProd …
    -/
    exact fun _ => h
    /-
      🎉 no goals
    -/


theorem elim_preimage_pi [DecidableEq ι] {l : List ι} (hnd : l.Nodup) (h : ∀ i, i ∈ l)
    (t : ∀ i, Set (α i)) : TProd.elim' h ⁻¹' pi univ t = Set.tprod l t := by
  have h2 : { i | i ∈ l } = univ := by
    ext i
    simp [h]
  /-
    ι : Type u
    α : ι → Type v
    inst✝ : DecidableEq ι
    l : List ι
    hnd : l.Nodup
    h : ∀ (i : ι), Membership.mem l i
    t : (i : ι) → Set (α i)
    h2 : Eq (setOf fun i => Membership.mem l i) Set.univ
    ⊢ Eq (Set.preimage (List.TProd.elim' h) (Set.univ.pi t)) (Set.tprod l t)
  -/
  rw [← h2, ← mk_preimage_tprod, preimage_preimage]
  /-
    ι : Type u
    α : ι → Type v
    inst✝ : DecidableEq ι
    l : List ι
    hnd : l.Nodup
    h : ∀ (i : ι), Membership.mem l i
    t : (i : ι) → Set (α i)
    h2 : Eq (setOf fun i => Membership.mem l i) Set.univ
    ⊢ Eq (Set.preimage (fun x => List.TProd.mk l (List.TProd.elim' h x)) (Set.tpro …
  -/
  simp only [TProd.mk_elim hnd h]
  /-
    ι : Type u
    α : ι → Type v
    inst✝ : DecidableEq ι
    l : List ι
    hnd : l.Nodup
    h : ∀ (i : ι), Membership.mem l i
    t : (i : ι) → Set (α i)
    h2 : Eq (setOf fun i => Membership.mem l i) Set.univ
    ⊢ Eq (Set.preimage (fun x => x) (Set.tprod l t)) (Set.tprod l t)
  -/
  dsimp
  /-
    🎉 no goals
  -/


