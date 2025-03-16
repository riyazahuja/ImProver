@[inherit_doc]
infixr:67 " ::ᵥ " => Vector.cons


instance [Inhabited α] : Inhabited (Vector α n) :=
  ⟨ofFn default⟩


theorem toList_injective : Function.Injective (@toList α n) :=
  Subtype.val_injective


/-- Two `v w : Vector α n` are equal iff they are equal at every single index. -/
@[ext]
theorem ext : ∀ {v w : Vector α n} (_ : ∀ m : Fin n, Vector.get v m = Vector.get w m), v = w
  | ⟨v, hv⟩, ⟨w, hw⟩, h =>
                                 /-
                                   α : Type u_1
                                   n : Nat
                                   v : List α
                                   hv : Eq v.length n
                                   w : List α
                                   hw : Eq w.length n
                                   h : ∀ (m : Fin n), Eq (List.Vector.get ⟨v, hv⟩ m) (List.Vector.get ⟨w, hw⟩ m)
                                   ⊢ Eq (↑⟨v, hv⟩).length (↑⟨w, hw⟩).length
                                 -/
    Subtype.eq (List.ext_get (by rw [hv, hw]) fun m hm _ => h ⟨m, hv ▸ hm⟩)
                                 /-
                                   🎉 no goals
                                 -/


/-- The empty `Vector` is a `Subsingleton`. -/
instance zero_subsingleton : Subsingleton (Vector α 0) :=
  ⟨fun _ _ => Vector.ext fun m => Fin.elim0 m⟩


@[simp]
theorem cons_val (a : α) : ∀ v : Vector α n, (a ::ᵥ v).val = a :: v.val
  | ⟨_, _⟩ => rfl


theorem eq_cons_iff (a : α) (v : Vector α n.succ) (v' : Vector α n) :
    v = a ::ᵥ v' ↔ v.head = a ∧ v.tail = v' :=
  ⟨fun h => h.symm ▸ ⟨head_cons a v', tail_cons a v'⟩, fun h =>
                                             /-
                                               α : Type u_1
                                               n : Nat
                                               a : α
                                               v : List.Vector α n.succ
                                               v' : List.Vector α n
                                               h : And (Eq v.head a) (Eq v.tail v')
                                               ⊢ Eq (List.Vector.cons v.head v.tail) (List.Vector.cons a v')
                                             -/
    _root_.trans (cons_head_tail v).symm (by rw [h.1, h.2])⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem ne_cons_iff (a : α) (v : Vector α n.succ) (v' : Vector α n) :
                                                  /-
                                                    α : Type u_1
                                                    n : Nat
                                                    a : α
                                                    v : List.Vector α n.succ
                                                    v' : List.Vector α n
                                                    ⊢ Iff (Ne v (List.Vector.cons a v')) (Or (Ne v.head a) (Ne v.tail v'))
                                                  -/
    v ≠ a ::ᵥ v' ↔ v.head ≠ a ∨ v.tail ≠ v' := by rw [Ne, eq_cons_iff a v v', not_and_or]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem exists_eq_cons (v : Vector α n.succ) : ∃ (a : α) (as : Vector α n), v = a ::ᵥ as :=
  ⟨v.head, v.tail, (eq_cons_iff v.head v v.tail).2 ⟨rfl, rfl⟩⟩


@[simp]
theorem toList_ofFn : ∀ {n} (f : Fin n → α), toList (ofFn f) = List.ofFn f
               /-
                 α : Type u_1
                 f : Fin 0 → α
                 ⊢ Eq (List.Vector.ofFn f).toList (List.ofFn f)
               -/
  | 0, f => by rw [ofFn, List.ofFn_zero, toList, nil]
               /-
                 🎉 no goals
               -/
                   /-
                     α : Type u_1
                     n : Nat
                     f : Fin (HAdd.hAdd n 1) → α
                     ⊢ Eq (List.Vector.ofFn f).toList (List.ofFn f)
                   -/
  | n + 1, f => by rw [ofFn, List.ofFn_succ, toList_cons, toList_ofFn]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mk_toList : ∀ (v : Vector α n) (h), (⟨toList v, h⟩ : Vector α n) = v
  | ⟨_, _⟩, _ => rfl



@[simp] theorem length_val (v : Vector α n) : v.val.length = n := v.2


@[simp]
theorem pmap_cons {p : α → Prop} (f : (a : α) → p a → β) (a : α) (v : Vector α n)
    (hp : ∀ x ∈ (cons a v).toList, p x) :
    (cons a v).pmap f hp = cons (f a (by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        σ : Type u_4
        φ : Type u_5
        m n : Nat
        p : α → Prop
        f : (a : α) → p a → β
        a : α
        v : List.Vector α n
        hp : ∀ (x : α), Membership.mem (List.Vector.cons a v).toList x → p x
        ⊢ p a
      -/
      simp only [Nat.succ_eq_add_one, toList_cons, List.mem_cons, forall_eq_or_imp] at hp
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        σ : Type u_4
        φ : Type u_5
        m n : Nat
        p : α → Prop
        f : (a : α) → p a → β
        a : α
        v : List.Vector α n
        hp : And (p a) (∀ (a : α), Membership.mem v.toList a → p a)
        ⊢ p a
      -/
      exact hp.1))
      /-
        🎉 no goals
      -/
      (v.pmap f (by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          σ : Type u_4
          φ : Type u_5
          m n : Nat
          p : α → Prop
          f : (a : α) → p a → β
          a : α
          v : List.Vector α n
          hp : ∀ (x : α), Membership.mem (List.Vector.cons a v).toList x → p x
          ⊢ ∀ (x : α), Membership.mem v.toList x → p x
        -/
        simp only [Nat.succ_eq_add_one, toList_cons, List.mem_cons, forall_eq_or_imp] at hp
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          σ : Type u_4
          φ : Type u_5
          m n : Nat
          p : α → Prop
          f : (a : α) → p a → β
          a : α
          v : List.Vector α n
          hp : And (p a) (∀ (a : α), Membership.mem v.toList a → p a)
          ⊢ ∀ (x : α), Membership.mem v.toList x → p x
        -/
        exact hp.2)) := rfl
        /-
          🎉 no goals
        -/


/-- Opposite direction of `Vector.pmap_cons` -/
theorem pmap_cons' {p : α → Prop} (f : (a : α) → p a → β) (a : α) (v : Vector α n)
    (ha : p a) (hp : ∀ x ∈ v.toList, p x) :
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          γ : Type u_3
                                                          σ : Type u_4
                                                          φ : Type u_5
                                                          m n : Nat
                                                          p : α → Prop
                                                          f : (a : α) → p a → β
                                                          a : α
                                                          v : List.Vector α n
                                                          ha : p a
                                                          hp : ∀ (x : α), Membership.mem v.toList x → p x
                                                          ⊢ ∀ (x : α), Membership.mem (List.Vector.cons a v).toList x → p x
                                                        -/
    cons (f a ha) (v.pmap f hp) = (cons a v).pmap f (by simpa [ha]) := rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem toList_map {β : Type*} (v : Vector α n) (f : α → β) :
                                            /-
                                              α : Type u_1
                                              n : Nat
                                              β : Type u_6
                                              v : List.Vector α n
                                              f : α → β
                                              ⊢ Eq (List.Vector.map f v).toList (List.map f v.toList)
                                            -/
    (v.map f).toList = v.toList.map f := by cases v; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem head_map {β : Type*} (v : Vector α (n + 1)) (f : α → β) : (v.map f).head = f v.head := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    v : List.Vector α (HAdd.hAdd n 1)
    f : α → β
    ⊢ Eq (List.Vector.map f v).head (f v.head)
  -/
  obtain ⟨a, v', h⟩ := Vector.exists_eq_cons v
  /-
    case intro.intro
    α : Type u_1
    n : Nat
    β : Type u_6
    v : List.Vector α (HAdd.hAdd n 1)
    f : α → β
    a : α
    v' : List.Vector α n
    h : Eq v (List.Vector.cons a v')
    ⊢ Eq (List.Vector.map f v).head (f v.head)
  -/
  rw [h, map_cons, head_cons, head_cons]
  /-
    🎉 no goals
  -/


@[simp]
theorem tail_map {β : Type*} (v : Vector α (n + 1)) (f : α → β) :
    (v.map f).tail = v.tail.map f := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    v : List.Vector α (HAdd.hAdd n 1)
    f : α → β
    ⊢ Eq (List.Vector.map f v).tail (List.Vector.map f v.tail)
  -/
  obtain ⟨a, v', h⟩ := Vector.exists_eq_cons v
  /-
    case intro.intro
    α : Type u_1
    n : Nat
    β : Type u_6
    v : List.Vector α (HAdd.hAdd n 1)
    f : α → β
    a : α
    v' : List.Vector α n
    h : Eq v (List.Vector.cons a v')
    ⊢ Eq (List.Vector.map f v).tail (List.Vector.map f v.tail)
  -/
  rw [h, map_cons, tail_cons, tail_cons]
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem_map {β : Type*} (v : Vector α n) (f : α → β) {i : ℕ} (hi : i < n) :
    (v.map f)[i] = f v[i] := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    v : List.Vector α n
    f : α → β
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (GetElem.getElem (List.Vector.map f v) i hi) (f (GetElem.getElem v i hi))
  -/
  simp only [getElem_def, toList_map, List.getElem_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem toList_pmap {p : α → Prop} (f : (a : α) → p a → β) (v : Vector α n)
    (hp : ∀ x ∈ v.toList, p x) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      n : Nat
                                                      p : α → Prop
                                                      f : (a : α) → p a → β
                                                      v : List.Vector α n
                                                      hp : ∀ (x : α), Membership.mem v.toList x → p x
                                                      ⊢ Eq (List.Vector.pmap f v hp).toList (List.pmap f v.toList hp)
                                                    -/
    (v.pmap f hp).toList = v.toList.pmap f hp := by cases v; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem head_pmap {p : α → Prop} (f : (a : α) → p a → β) (v : Vector α (n + 1))
    (hp : ∀ x ∈ v.toList, p x) :
    (v.pmap f hp).head = f v.head (hp _ <| by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        σ : Type u_4
        φ : Type u_5
        m n : Nat
        p : α → Prop
        f : (a : α) → p a → β
        v : List.Vector α (HAdd.hAdd n 1)
        hp : ∀ (x : α), Membership.mem v.toList x → p x
        ⊢ Membership.mem v.toList v.head
      -/
      rw [← cons_head_tail v, toList_cons, head_cons, List.mem_cons]; exact .inl rfl) := by
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    p : α → Prop
    f : (a : α) → p a → β
    v : List.Vector α (HAdd.hAdd n 1)
    hp : ∀ (x : α), Membership.mem v.toList x → p x
    ⊢ Eq (List.Vector.pmap f v hp).head (f v.head ⋯)
  -/
  obtain ⟨a, v', h⟩ := Vector.exists_eq_cons v
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    n : Nat
    p : α → Prop
    f : (a : α) → p a → β
    v : List.Vector α (HAdd.hAdd n 1)
    hp : ∀ (x : α), Membership.mem v.toList x → p x
    a : α
    v' : List.Vector α n
    h : Eq v (List.Vector.cons a v')
    ⊢ Eq (List.Vector.pmap f v hp).head (f v.head ⋯)
  -/
  simp_rw [h, pmap_cons, head_cons]
  /-
    🎉 no goals
  -/


@[simp]
theorem tail_pmap {p : α → Prop} (f : (a : α) → p a → β) (v : Vector α (n + 1))
    (hp : ∀ x ∈ v.toList, p x) :
    (v.pmap f hp).tail = v.tail.pmap f (fun x hx ↦ hp _ <| by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        σ : Type u_4
        φ : Type u_5
        m n : Nat
        p : α → Prop
        f : (a : α) → p a → β
        v : List.Vector α (HAdd.hAdd n 1)
        hp : ∀ (x : α), Membership.mem v.toList x → p x
        x : α
        hx : Membership.mem v.tail.toList x
        ⊢ Membership.mem v.toList x
      -/
      rw [← cons_head_tail v, toList_cons, List.mem_cons]; exact .inr hx) := by
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    p : α → Prop
    f : (a : α) → p a → β
    v : List.Vector α (HAdd.hAdd n 1)
    hp : ∀ (x : α), Membership.mem v.toList x → p x
    ⊢ Eq (List.Vector.pmap f v hp).tail (List.Vector.pmap f v.tail ⋯)
  -/
  obtain ⟨a, v', h⟩ := Vector.exists_eq_cons v
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    n : Nat
    p : α → Prop
    f : (a : α) → p a → β
    v : List.Vector α (HAdd.hAdd n 1)
    hp : ∀ (x : α), Membership.mem v.toList x → p x
    a : α
    v' : List.Vector α n
    h : Eq v (List.Vector.cons a v')
    ⊢ Eq (List.Vector.pmap f v hp).tail (List.Vector.pmap f v.tail ⋯)
  -/
  simp_rw [h, pmap_cons, tail_cons]
  /-
    🎉 no goals
  -/


@[simp]
theorem getElem_pmap {p : α → Prop} (f : (a : α) → p a → β) (v : Vector α n)
    (hp : ∀ x ∈ v.toList, p x) {i : ℕ} (hi : i < n) :
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          γ : Type u_3
                                          σ : Type u_4
                                          φ : Type u_5
                                          m n : Nat
                                          p : α → Prop
                                          f : (a : α) → p a → β
                                          v : List.Vector α n
                                          hp : ∀ (x : α), Membership.mem v.toList x → p x
                                          i : Nat
                                          hi : LT.lt i n
                                          ⊢ Membership.mem v.toList (GetElem.getElem v i hi)
                                        -/
    (v.pmap f hp)[i] = f v[i] (hp _ (by simp [getElem_def, List.getElem_mem])) := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    α : Type u_1
    β : Type u_2
    n : Nat
    p : α → Prop
    f : (a : α) → p a → β
    v : List.Vector α n
    hp : ∀ (x : α), Membership.mem v.toList x → p x
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (GetElem.getElem (List.Vector.pmap f v hp) i hi) (f (GetElem.getElem v i  …
  -/
  simp only [getElem_def, toList_pmap, List.getElem_pmap]
  /-
    🎉 no goals
  -/


theorem get_eq_get_toList (v : Vector α n) (i : Fin n) :
    v.get i = v.toList.get (Fin.cast v.toList_length.symm i) :=
  rfl


@[deprecated (since := "2024-12-20")]
alias get_eq_get := get_eq_get_toList


@[simp]
theorem get_replicate (a : α) (i : Fin n) : (Vector.replicate n a).get i = a := by
  /-
    α : Type u_1
    n : Nat
    a : α
    i : Fin n
    ⊢ Eq ((List.Vector.replicate n a).get i) a
  -/
  apply List.getElem_replicate
  /-
    🎉 no goals
  -/


@[simp]
theorem get_map {β : Type*} (v : Vector α n) (f : α → β) (i : Fin n) :
    (v.map f).get i = f (v.get i) := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    v : List.Vector α n
    f : α → β
    i : Fin n
    ⊢ Eq ((List.Vector.map f v).get i) (f (v.get i))
  -/
  cases v; simp [Vector.map, get_eq_get_toList]
           /-
             🎉 no goals
           -/


@[simp]
theorem map₂_nil (f : α → β → γ) : Vector.map₂ f nil nil = nil :=
  rfl


@[simp]
theorem map₂_cons (hd₁ : α) (tl₁ : Vector α n) (hd₂ : β) (tl₂ : Vector β n) (f : α → β → γ) :
    Vector.map₂ f (hd₁ ::ᵥ tl₁) (hd₂ ::ᵥ tl₂) = f hd₁ hd₂ ::ᵥ (Vector.map₂ f tl₁ tl₂) :=
  rfl


@[simp]
theorem get_ofFn {n} (f : Fin n → α) (i) : get (ofFn f) i = f i := by
  /-
    α : Type u_1
    n : Nat
    f : Fin n → α
    i : Fin n
    ⊢ Eq ((List.Vector.ofFn f).get i) (f i)
  -/
  conv_rhs => erw [← List.get_ofFn f ⟨i, by simp⟩]
  /-
    α : Type u_1
    n : Nat
    f : Fin n → α
    i : Fin n
    ⊢ Eq ((List.Vector.ofFn f).get i) ((List.ofFn f).get ⟨↑i, ⋯⟩)
  -/
  simp only [get_eq_get_toList]
  /-
    α : Type u_1
    n : Nat
    f : Fin n → α
    i : Fin n
    ⊢ Eq ((List.Vector.ofFn f).toList.get (Fin.cast ⋯ i)) ((List.ofFn f).get ⟨↑i,  …
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
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
  congr <;> simp [Fin.heq_ext_iff]
            /-
              🎉 no goals
            -/


@[simp]
theorem ofFn_get (v : Vector α n) : ofFn (get v) = v := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α n
    ⊢ Eq (List.Vector.ofFn v.get) v
  -/
  rcases v with ⟨l, rfl⟩
  /-
    case mk
    α : Type u_1
    l : List α
    ⊢ Eq (List.Vector.ofFn (List.Vector.get ⟨l, ⋯⟩)) ⟨l, ⋯⟩
  -/
  apply toList_injective
  /-
    case mk.a
    α : Type u_1
    l : List α
    ⊢ Eq (List.Vector.ofFn (List.Vector.get ⟨l, ⋯⟩)).toList (List.Vector.toList ⟨l …
  -/
  dsimp
  /-
    case mk.a
    α : Type u_1
    l : List α
    ⊢ Eq (List.Vector.ofFn (List.Vector.get ⟨l, ⋯⟩)).toList l
  -/
  simpa only [toList_ofFn] using List.ofFn_get _
  /-
    🎉 no goals
  -/


/-- The natural equivalence between length-`n` vectors and functions from `Fin n`. -/
def _root_.Equiv.vectorEquivFin (α : Type*) (n : ℕ) : Vector α n ≃ (Fin n → α) :=
  ⟨Vector.get, Vector.ofFn, Vector.ofFn_get, fun f => funext <| Vector.get_ofFn f⟩


                                                                          /-
                                                                            α : Type u_1
                                                                            β : Type u_2
                                                                            γ : Type u_3
                                                                            σ : Type u_4
                                                                            φ : Type u_5
                                                                            m n : Nat
                                                                            x : List.Vector α n
                                                                            i : Fin (HSub.hSub n 1)
                                                                            ⊢ LT.lt (HAdd.hAdd (↑i) 1) n
                                                                          -/
theorem get_tail (x : Vector α n) (i) : x.tail.get i = x.get ⟨i.1 + 1, by omega⟩ := by
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  /-
    α : Type u_1
    n : Nat
    x : List.Vector α n
    i : Fin (HSub.hSub n 1)
    ⊢ Eq (x.tail.get i) (x.get ⟨HAdd.hAdd (↑i) 1, ⋯⟩)
  -/
  cases' i with i ih; dsimp
  /-
    case mk
    α : Type u_1
    n : Nat
    x : List.Vector α n
    i : Nat
    ih : LT.lt i (HSub.hSub n 1)
    ⊢ Eq (x.tail.get ⟨i, ih⟩) (x.get ⟨HAdd.hAdd i 1, ⋯⟩)
  -/
  rcases x with ⟨_ | _, h⟩ <;> try rfl
                               /-
                                 🎉 no goals
                               -/
  /-
    case mk.mk.nil
    α : Type u_1
    n i : Nat
    ih : LT.lt i (HSub.hSub n 1)
    h : Eq List.nil.length n
    ⊢ Eq ((List.Vector.tail ⟨List.nil, h⟩).get ⟨i, ih⟩) (List.Vector.get ⟨List.nil …
  -/
  rw [List.length] at h
  /-
    case mk.mk.nil
    α : Type u_1
    n i : Nat
    ih : LT.lt i (HSub.hSub n 1)
    h✝ : Eq List.nil.length n
    h : Eq 0 n
    ⊢ Eq ((List.Vector.tail ⟨List.nil, h✝⟩).get ⟨i, ih⟩) (List.Vector.get ⟨List.ni …
  -/
  rw [← h] at ih
  /-
    case mk.mk.nil
    α : Type u_1
    n i : Nat
    ih✝ : LT.lt i (HSub.hSub n 1)
    ih : LT.lt i (HSub.hSub 0 1)
    h✝ : Eq List.nil.length n
    h : Eq 0 n
    ⊢ Eq ((List.Vector.tail ⟨List.nil, h✝⟩).get ⟨i, ih✝⟩) (List.Vector.get ⟨List.n …
  -/
  contradiction
  /-
    🎉 no goals
  -/


@[simp]
theorem get_tail_succ : ∀ (v : Vector α n.succ) (i : Fin n), get (tail v) i = get v i.succ
                              /-
                                α : Type u_1
                                n : Nat
                                a : α
                                l : List α
                                e : Eq (List.cons a l).length n.succ
                                i : Nat
                                h : LT.lt i n
                                ⊢ Eq ((List.Vector.tail ⟨List.cons a l, e⟩).get ⟨i, h⟩) (List.Vector.get ⟨List …
                              -/
  | ⟨a :: l, e⟩, ⟨i, h⟩ => by simp [get_eq_get_toList]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem tail_val : ∀ v : Vector α n.succ, v.tail.val = v.val.tail
  | ⟨_ :: _, _⟩ => rfl


/-- The `tail` of a `nil` vector is `nil`. -/
@[simp]
theorem tail_nil : (@nil α).tail = nil :=
  rfl


/-- The `tail` of a vector made up of one element is `nil`. -/
@[simp]
theorem singleton_tail : ∀ (v : Vector α 1), v.tail = Vector.nil
  | ⟨[_], _⟩ => rfl


@[simp]
theorem tail_ofFn {n : ℕ} (f : Fin n.succ → α) : tail (ofFn f) = ofFn fun i => f i.succ :=
  (ofFn_get _).symm.trans <| by
    /-
      α : Type u_1
      n : Nat
      f : Fin n.succ → α
      ⊢ Eq (List.Vector.ofFn (List.Vector.ofFn f).tail.get) (List.Vector.ofFn fun i  …
    -/
    congr
    /-
      case e_a
      α : Type u_1
      n : Nat
      f : Fin n.succ → α
      ⊢ Eq (List.Vector.ofFn f).tail.get fun i => f i.succ
    -/
    funext i
    /-
      case e_a.h
      α : Type u_1
      n : Nat
      f : Fin n.succ → α
      i : Fin (HSub.hSub n.succ 1)
      ⊢ Eq ((List.Vector.ofFn f).tail.get i) (f i.succ)
    -/
    rw [get_tail, get_ofFn]
    /-
      case e_a.h
      α : Type u_1
      n : Nat
      f : Fin n.succ → α
      i : Fin (HSub.hSub n.succ 1)
      ⊢ Eq (f ⟨HAdd.hAdd (↑i) 1, ⋯⟩) (f i.succ)
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem toList_empty (v : Vector α 0) : v.toList = [] :=
  List.length_eq_zero.mp v.2


/-- The list that makes up a `Vector` made up of a single element,
retrieved via `toList`, is equal to the list of that single element. -/
@[simp]
theorem toList_singleton (v : Vector α 1) : v.toList = [v.head] := by
  /-
    α : Type u_1
    v : List.Vector α 1
    ⊢ Eq v.toList (List.cons v.head List.nil)
  -/
  rw [← v.cons_head_tail]
  /-
    α : Type u_1
    v : List.Vector α 1
    ⊢ Eq (List.Vector.cons v.head v.tail).toList (List.cons (List.Vector.cons v.he …
  -/
  simp only [toList_cons, toList_nil, head_cons, eq_self_iff_true, and_self_iff, singleton_tail]
  /-
    🎉 no goals
  -/


@[simp]
theorem empty_toList_eq_ff (v : Vector α (n + 1)) : v.toList.isEmpty = false :=
  match v with
  | ⟨_ :: _, _⟩ => rfl


theorem not_empty_toList (v : Vector α (n + 1)) : ¬v.toList.isEmpty := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α (HAdd.hAdd n 1)
    ⊢ Not (Eq v.toList.isEmpty Bool.true)
  -/
  simp only [empty_toList_eq_ff, Bool.coe_sort_false, not_false_iff]
  /-
    🎉 no goals
  -/


/-- Mapping under `id` does not change a vector. -/
@[simp]
theorem map_id {n : ℕ} (v : Vector α n) : Vector.map id v = v :=
                    /-
                      α : Type u_1
                      n : Nat
                      v : List.Vector α n
                      ⊢ Eq (List.Vector.map id v).toList v.toList
                    -/
  Vector.eq _ _ (by simp only [List.map_id, Vector.toList_map])
                    /-
                      🎉 no goals
                    -/


theorem nodup_iff_injective_get {v : Vector α n} : v.toList.Nodup ↔ Function.Injective v.get := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α n
    ⊢ Iff v.toList.Nodup (Function.Injective v.get)
  -/
  cases' v with l hl
  /-
    case mk
    α : Type u_1
    n : Nat
    l : List α
    hl : Eq l.length n
    ⊢ Iff (List.Vector.toList ⟨l, hl⟩).Nodup (Function.Injective (List.Vector.get  …
  -/
  subst hl
  /-
    case mk
    α : Type u_1
    l : List α
    ⊢ Iff (List.Vector.toList ⟨l, ⋯⟩).Nodup (Function.Injective (List.Vector.get ⟨ …
  -/
  exact List.nodup_iff_injective_get
  /-
    🎉 no goals
  -/


theorem head?_toList : ∀ v : Vector α n.succ, (toList v).head? = some (head v)
  | ⟨_ :: _, _⟩ => rfl


/-- Reverse a vector. -/
def reverse (v : Vector α n) : Vector α n :=
                        /-
                          α : Type u_1
                          β : Type u_2
                          γ : Type u_3
                          σ : Type u_4
                          φ : Type u_5
                          m n : Nat
                          v : List.Vector α n
                          ⊢ Eq v.toList.reverse.length n
                        -/
  ⟨v.toList.reverse, by simp⟩
                        /-
                          🎉 no goals
                        -/


/-- The `List` of a vector after a `reverse`, retrieved by `toList` is equal
to the `List.reverse` after retrieving a vector's `toList`. -/
theorem toList_reverse {v : Vector α n} : v.reverse.toList = v.toList.reverse :=
  rfl


@[simp]
theorem reverse_reverse {v : Vector α n} : v.reverse.reverse = v := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α n
    ⊢ Eq v.reverse.reverse v
  -/
  cases v
  /-
    case mk
    α : Type u_1
    n : Nat
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq (List.Vector.reverse ⟨val✝, property✝⟩).reverse ⟨val✝, property✝⟩
  -/
  simp [Vector.reverse]
  /-
    🎉 no goals
  -/


@[simp]
theorem get_zero : ∀ v : Vector α n.succ, get v 0 = head v
  | ⟨_ :: _, _⟩ => rfl


@[simp]
theorem head_ofFn {n : ℕ} (f : Fin n.succ → α) : head (ofFn f) = f 0 := by
  /-
    α : Type u_1
    n : Nat
    f : Fin n.succ → α
    ⊢ Eq (List.Vector.ofFn f).head (f 0)
  -/
  rw [← get_zero, get_ofFn]
  /-
    🎉 no goals
  -/


                                                                           /-
                                                                             α : Type u_1
                                                                             n : Nat
                                                                             a : α
                                                                             v : List.Vector α n
                                                                             ⊢ Eq ((List.Vector.cons a v).get 0) a
                                                                           -/
theorem get_cons_zero (a : α) (v : Vector α n) : get (a ::ᵥ v) 0 = a := by simp [get_zero]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- Accessing the nth element of a vector made up
of one element `x : α` is `x` itself. -/
@[simp]
theorem get_cons_nil : ∀ {ix : Fin 1} (x : α), get (x ::ᵥ nil) ix = x
  | ⟨0, _⟩, _ => rfl


@[simp]
theorem get_cons_succ (a : α) (v : Vector α n) (i : Fin n) : get (a ::ᵥ v) i.succ = get v i := by
  /-
    α : Type u_1
    n : Nat
    a : α
    v : List.Vector α n
    i : Fin n
    ⊢ Eq ((List.Vector.cons a v).get i.succ) (v.get i)
  -/
  rw [← get_tail_succ, tail_cons]
  /-
    🎉 no goals
  -/


/-- The last element of a `Vector`, given that the vector is at least one element. -/
def last (v : Vector α (n + 1)) : α :=
  v.get (Fin.last n)


/-- The last element of a `Vector`, given that the vector is at least one element. -/
theorem last_def {v : Vector α (n + 1)} : v.last = v.get (Fin.last n) :=
  rfl


/-- The `last` element of a vector is the `head` of the `reverse` vector. -/
theorem reverse_get_zero {v : Vector α (n + 1)} : v.reverse.head = v.last := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α (HAdd.hAdd n 1)
    ⊢ Eq v.reverse.head v.last
  -/
  rw [← get_zero, last_def, get_eq_get_toList, get_eq_get_toList]
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α (HAdd.hAdd n 1)
    ⊢ Eq (v.reverse.toList.get (Fin.cast ⋯ 0)) (v.toList.get (Fin.cast ⋯ (Fin.last …
  -/
  simp_rw [toList_reverse]
  rw [List.get_eq_getElem, List.get_eq_getElem, ← Option.some_inj, Fin.cast, Fin.cast,
    ← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem, List.getElem?_reverse]
    /-
      α : Type u_1
      n : Nat
      v : List.Vector α (HAdd.hAdd n 1)
      ⊢ Eq (GetElem?.getElem? v.toList (HSub.hSub (HSub.hSub v.toList.length 1) ↑⟨↑0 …
    -/
  · congr
    /-
      case e_a
      α : Type u_1
      n : Nat
      v : List.Vector α (HAdd.hAdd n 1)
      ⊢ Eq (HSub.hSub (HSub.hSub v.toList.length 1) ↑⟨↑0, ⋯⟩) ↑⟨↑(Fin.last n), ⋯⟩
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      n : Nat
      v : List.Vector α (HAdd.hAdd n 1)
      ⊢ LT.lt (↑⟨↑0, ⋯⟩) v.toList.length
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- Construct a `Vector β (n + 1)` from a `Vector α n` by scanning `f : β → α → β`
from the "left", that is, from 0 to `Fin.last n`, using `b : β` as the starting value.
-/
def scanl : Vector β (n + 1) :=
                               /-
                                 α : Type u_1
                                 β✝ : Type u_2
                                 γ : Type u_3
                                 σ : Type u_4
                                 φ : Type u_5
                                 m n : Nat
                                 β : Type u_6
                                 f : β → α → β
                                 b : β
                                 v : List.Vector α n
                                 ⊢ Eq (List.scanl f b v.toList).length (HAdd.hAdd n 1)
                               -/
  ⟨List.scanl f b v.toList, by rw [List.length_scanl, toList_length]⟩
                               /-
                                 🎉 no goals
                               -/


/-- Providing an empty vector to `scanl` gives the starting value `b : β`. -/
@[simp]
theorem scanl_nil : scanl f b nil = b ::ᵥ nil :=
  rfl


/-- The recursive step of `scanl` splits a vector `x ::ᵥ v : Vector α (n + 1)`
into the provided starting value `b : β` and the recursed `scanl`
`f b x : β` as the starting value.

This lemma is the `cons` version of `scanl_get`.
-/
@[simp]
theorem scanl_cons (x : α) : scanl f b (x ::ᵥ v) = b ::ᵥ scanl f (f b x) v := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    f : β → α → β
    b : β
    v : List.Vector α n
    x : α
    ⊢ Eq (List.Vector.scanl f b (List.Vector.cons x v)) (List.Vector.cons b (List. …
  -/
  simp only [scanl, toList_cons, List.scanl]; dsimp
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    f : β → α → β
    b : β
    v : List.Vector α n
    x : α
    ⊢ Eq ⟨List.cons b (List.scanl f (f b x) ↑v), ⋯⟩ (List.Vector.cons b ⟨List.scan …
  -/
  simp only [cons]; rfl
                    /-
                      🎉 no goals
                    -/


/-- The underlying `List` of a `Vector` after a `scanl` is the `List.scanl`
of the underlying `List` of the original `Vector`.
-/
@[simp]
theorem scanl_val : ∀ {v : Vector α n}, (scanl f b v).val = List.scanl f b v.val
  | _ => rfl


/-- The `toList` of a `Vector` after a `scanl` is the `List.scanl`
of the `toList` of the original `Vector`.
-/
@[simp]
theorem toList_scanl : (scanl f b v).toList = List.scanl f b v.toList :=
  rfl


/-- The recursive step of `scanl` splits a vector made up of a single element
`x ::ᵥ nil : Vector α 1` into a `Vector` of the provided starting value `b : β`
and the mapped `f b x : β` as the last value.
-/
@[simp]
theorem scanl_singleton (v : Vector α 1) : scanl f b v = b ::ᵥ f b v.head ::ᵥ nil := by
  /-
    α : Type u_1
    β : Type u_6
    f : β → α → β
    b : β
    v : List.Vector α 1
    ⊢ Eq (List.Vector.scanl f b v) (List.Vector.cons b (List.Vector.cons (f b v.he …
  -/
  rw [← cons_head_tail v]
  /-
    α : Type u_1
    β : Type u_6
    f : β → α → β
    b : β
    v : List.Vector α 1
    ⊢ Eq (List.Vector.scanl f b (List.Vector.cons v.head v.tail)) (List.Vector.con …
  -/
  simp only [scanl_cons, scanl_nil, head_cons, singleton_tail]
  /-
    🎉 no goals
  -/


/-- The first element of `scanl` of a vector `v : Vector α n`,
retrieved via `head`, is the starting value `b : β`.
-/
@[simp]
theorem scanl_head : (scanl f b v).head = b := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    f : β → α → β
    b : β
    v : List.Vector α n
    ⊢ Eq (List.Vector.scanl f b v).head b
  -/
  cases n
    /-
      case zero
      α : Type u_1
      β : Type u_6
      f : β → α → β
      b : β
      v : List.Vector α 0
      ⊢ Eq (List.Vector.scanl f b v).head b
    -/
  · have : v = nil := by simp only [eq_iff_true_of_subsingleton]
    /-
      case zero
      α : Type u_1
      β : Type u_6
      f : β → α → β
      b : β
      v : List.Vector α 0
      this : Eq v List.Vector.nil
      ⊢ Eq (List.Vector.scanl f b v).head b
    -/
    simp only [this, scanl_nil, head_cons]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      β : Type u_6
      f : β → α → β
      b : β
      n✝ : Nat
      v : List.Vector α (HAdd.hAdd n✝ 1)
      ⊢ Eq (List.Vector.scanl f b v).head b
    -/
  · rw [← cons_head_tail v]
    simp only [← get_zero, get_eq_get_toList, toList_scanl, toList_cons, List.scanl, Fin.val_zero,
      List.get]


/-- For an index `i : Fin n`, the nth element of `scanl` of a
vector `v : Vector α n` at `i.succ`, is equal to the application
function `f : β → α → β` of the `castSucc i` element of
`scanl f b v` and `get v i`.

This lemma is the `get` version of `scanl_cons`.
-/
@[simp]
theorem scanl_get (i : Fin n) :
    (scanl f b v).get i.succ = f ((scanl f b v).get (Fin.castSucc i)) (v.get i) := by
  /-
    α : Type u_1
    n : Nat
    β : Type u_6
    f : β → α → β
    b : β
    v : List.Vector α n
    i : Fin n
    ⊢ Eq ((List.Vector.scanl f b v).get i.succ) (f ((List.Vector.scanl f b v).get  …
  -/
  cases' n with n
    /-
      case zero
      α : Type u_1
      β : Type u_6
      f : β → α → β
      b : β
      v : List.Vector α 0
      i : Fin 0
      ⊢ Eq ((List.Vector.scanl f b v).get i.succ) (f ((List.Vector.scanl f b v).get  …
    -/
  · exact i.elim0
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    β : Type u_6
    f : β → α → β
    b : β
    n : Nat
    v : List.Vector α (HAdd.hAdd n 1)
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq ((List.Vector.scanl f b v).get i.succ) (f ((List.Vector.scanl f b v).get  …
  -/
  induction' n with n hn generalizing b
    /-
      case succ.zero
      α : Type u_1
      β : Type u_6
      f : β → α → β
      b : β
      v : List.Vector α (HAdd.hAdd 0 1)
      i : Fin (HAdd.hAdd 0 1)
      ⊢ Eq ((List.Vector.scanl f b v).get i.succ) (f ((List.Vector.scanl f b v).get  …
    -/
  · have i0 : i = 0 := Fin.eq_zero _
    /-
      case succ.zero
      α : Type u_1
      β : Type u_6
      f : β → α → β
      b : β
      v : List.Vector α (HAdd.hAdd 0 1)
      i : Fin (HAdd.hAdd 0 1)
      i0 : Eq i 0
      ⊢ Eq ((List.Vector.scanl f b v).get i.succ) (f ((List.Vector.scanl f b v).get  …
    -/
    simp [scanl_singleton, i0, get_zero]; simp [get_eq_get_toList, List.get]
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case succ.succ
      α : Type u_1
      β : Type u_6
      f : β → α → β
      n : Nat
      hn : ∀ (b : β) (v : List.Vector α (HAdd.hAdd n 1)) (i : Fin (HAdd.hAdd n 1)),  …
      b : β
      v : List.Vector α (HAdd.hAdd (HAdd.hAdd n 1) 1)
      i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq ((List.Vector.scanl f b v).get i.succ) (f ((List.Vector.scanl f b v).get  …
    -/
  · rw [← cons_head_tail v, scanl_cons, get_cons_succ]
    /-
      case succ.succ
      α : Type u_1
      β : Type u_6
      f : β → α → β
      n : Nat
      hn : ∀ (b : β) (v : List.Vector α (HAdd.hAdd n 1)) (i : Fin (HAdd.hAdd n 1)),  …
      b : β
      v : List.Vector α (HAdd.hAdd (HAdd.hAdd n 1) 1)
      i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq ((List.Vector.scanl f (f b v.head) v.tail).get i) (f ((List.Vector.cons b …
    -/
    refine Fin.cases ?_ ?_ i
      /-
        case succ.succ.refine_1
        α : Type u_1
        β : Type u_6
        f : β → α → β
        n : Nat
        hn : ∀ (b : β) (v : List.Vector α (HAdd.hAdd n 1)) (i : Fin (HAdd.hAdd n 1)),  …
        b : β
        v : List.Vector α (HAdd.hAdd (HAdd.hAdd n 1) 1)
        i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ Eq ((List.Vector.scanl f (f b v.head) v.tail).get 0) (f ((List.Vector.cons b …
      -/
    · simp only [get_zero, scanl_head, Fin.castSucc_zero, head_cons]
      /-
        🎉 no goals
      -/
      /-
        case succ.succ.refine_2
        α : Type u_1
        β : Type u_6
        f : β → α → β
        n : Nat
        hn : ∀ (b : β) (v : List.Vector α (HAdd.hAdd n 1)) (i : Fin (HAdd.hAdd n 1)),  …
        b : β
        v : List.Vector α (HAdd.hAdd (HAdd.hAdd n 1) 1)
        i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), Eq ((List.Vector.scanl f (f b v.head) v.tail).g …
      -/
    · intro i'
      /-
        case succ.succ.refine_2
        α : Type u_1
        β : Type u_6
        f : β → α → β
        n : Nat
        hn : ∀ (b : β) (v : List.Vector α (HAdd.hAdd n 1)) (i : Fin (HAdd.hAdd n 1)),  …
        b : β
        v : List.Vector α (HAdd.hAdd (HAdd.hAdd n 1) 1)
        i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
        i' : Fin (HAdd.hAdd n 1)
        ⊢ Eq ((List.Vector.scanl f (f b v.head) v.tail).get i'.succ) (f ((List.Vector. …
      -/
      simp only [hn, Fin.castSucc_fin_succ, get_cons_succ]
      /-
        🎉 no goals
      -/


/-- Monadic analog of `Vector.ofFn`.
Given a monadic function on `Fin n`, return a `Vector α n` inside the monad. -/
def mOfFn {m} [Monad m] {α : Type u} : ∀ {n}, (Fin n → m α) → m (Vector α n)
  | 0, _ => pure nil
  | _ + 1, f => do
    let a ← f 0
    let v ← mOfFn fun i => f i.succ
    pure (a ::ᵥ v)


theorem mOfFn_pure {m} [Monad m] [LawfulMonad m] {α} :
    ∀ {n} (f : Fin n → α), (@mOfFn m _ _ _ fun i => pure (f i)) = pure (ofFn f)
  | 0, _ => rfl
  | n + 1, f => by
    /-
      m : Type u_6 → Type u_7
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_6
      n : Nat
      f : Fin (HAdd.hAdd n 1) → α
      ⊢ Eq (List.Vector.mOfFn fun i => Pure.pure (f i)) (Pure.pure (List.Vector.ofFn …
    -/
    rw [mOfFn, @mOfFn_pure m _ _ _ n _, ofFn]
    /-
      m : Type u_6 → Type u_7
      inst✝¹ : Monad m
      inst✝ : LawfulMonad m
      α : Type u_6
      n : Nat
      f : Fin (HAdd.hAdd n 1) → α
      ⊢ Eq (Bind.bind (Pure.pure (f 0)) fun a => Bind.bind (Pure.pure (List.Vector.o …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Apply a monadic function to each component of a vector,
returning a vector inside the monad. -/
def mmap {m} [Monad m] {α} {β : Type u} (f : α → m β) : ∀ {n}, Vector α n → m (Vector β n)
  | 0, _ => pure nil
  | _ + 1, xs => do
    let h' ← f xs.head
    let t' ← mmap f xs.tail
    pure (h' ::ᵥ t')


@[simp]
theorem mmap_nil {m} [Monad m] {α β} (f : α → m β) : mmap f nil = pure nil :=
  rfl


@[simp]
theorem mmap_cons {m} [Monad m] {α β} (f : α → m β) (a) :
    ∀ {n} (v : Vector α n),
      mmap f (a ::ᵥ v) = do
        let h' ← f a
        let t' ← mmap f v
        pure (h' ::ᵥ t')
  | _, ⟨_, rfl⟩ => rfl


/--
Define `C v` by induction on `v : Vector α n`.

This function has two arguments: `nil` handles the base case on `C nil`,
and `cons` defines the inductive step using `∀ x : α, C w → C (x ::ᵥ w)`.

It is used as the default induction principle for the `induction` tactic.
-/
@[elab_as_elim, induction_eliminator]
def inductionOn {C : ∀ {n : ℕ}, Vector α n → Sort*} {n : ℕ} (v : Vector α n)
    (nil : C nil) (cons : ∀ {n : ℕ} {x : α} {w : Vector α n}, C w → C (x ::ᵥ w)) : C v := by
  -- Porting note: removed `generalizing`: already generalized
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    σ : Type u_4
    φ : Type u_5
    m n✝ : Nat
    C : {n : Nat} → List.Vector α n → Sort u_6
    n : Nat
    v : List.Vector α n
    nil : C List.Vector.nil
    cons : {n : Nat} → {x : α} → {w : List.Vector α n} → C w → C (List.Vector.cons …
    ⊢ C v
  -/
  induction' n with n ih
    /-
      case zero
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      C : {n : Nat} → List.Vector α n → Sort u_6
      nil : C List.Vector.nil
      cons : {n : Nat} → {x : α} → {w : List.Vector α n} → C w → C (List.Vector.cons …
      v : List.Vector α 0
      ⊢ C v
    -/
  · rcases v with ⟨_ | ⟨-, -⟩, - | -⟩
    /-
      case zero.mk.nil.refl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      C : {n : Nat} → List.Vector α n → Sort u_6
      nil : C List.Vector.nil
      cons : {n : Nat} → {x : α} → {w : List.Vector α n} → C w → C (List.Vector.cons …
      ⊢ C ⟨List.nil, ⋯⟩
    -/
    exact nil
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      C : {n : Nat} → List.Vector α n → Sort u_6
      nil : C List.Vector.nil
      cons : {n : Nat} → {x : α} → {w : List.Vector α n} → C w → C (List.Vector.cons …
      n : Nat
      ih : (v : List.Vector α n) → C v
      v : List.Vector α (HAdd.hAdd n 1)
      ⊢ C v
    -/
  · rcases v with ⟨_ | ⟨a, v⟩, v_property⟩
    /-
      case succ.mk.nil
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      C : {n : Nat} → List.Vector α n → Sort u_6
      nil : C List.Vector.nil
      cons : {n : Nat} → {x : α} → {w : List.Vector α n} → C w → C (List.Vector.cons …
      n : Nat
      ih : (v : List.Vector α n) → C v
      v_property : Eq List.nil.length (HAdd.hAdd n 1)
      ⊢ C ⟨List.nil, v_property⟩
    -/
    cases v_property
    /-
      case succ.mk.cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      C : {n : Nat} → List.Vector α n → Sort u_6
      nil : C List.Vector.nil
      cons : {n : Nat} → {x : α} → {w : List.Vector α n} → C w → C (List.Vector.cons …
      n : Nat
      ih : (v : List.Vector α n) → C v
      a : α
      v : List α
      v_property : Eq (List.cons a v).length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a v, v_property⟩
    -/
    exact cons (ih ⟨v, (add_left_inj 1).mp v_property⟩)
    /-
      🎉 no goals
    -/


@[simp]
theorem inductionOn_nil {C : ∀ {n : ℕ}, Vector α n → Sort*}
    (nil : C nil) (cons : ∀ {n : ℕ} {x : α} {w : Vector α n}, C w → C (x ::ᵥ w)) :
    Vector.nil.inductionOn nil cons = nil :=
  rfl


@[simp]
theorem inductionOn_cons {C : ∀ {n : ℕ}, Vector α n → Sort*} {n : ℕ} (x : α) (v : Vector α n)
    (nil : C nil) (cons : ∀ {n : ℕ} {x : α} {w : Vector α n}, C w → C (x ::ᵥ w)) :
    (x ::ᵥ v).inductionOn nil cons = cons (v.inductionOn nil cons : C v) :=
  rfl


/-- Define `C v w` by induction on a pair of vectors `v : Vector α n` and `w : Vector β n`. -/
@[elab_as_elim]
def inductionOn₂ {C : ∀ {n}, Vector α n → Vector β n → Sort*}
    (v : Vector α n) (w : Vector β n)
    (nil : C nil nil) (cons : ∀ {n a b} {x : Vector α n} {y}, C x y → C (a ::ᵥ x) (b ::ᵥ y)) :
    C v w := by
  -- Porting note: removed `generalizing`: already generalized
  /-
    α : Type u_1
    β✝ : Type u_2
    γ✝ : Type u_3
    σ : Type u_4
    φ : Type u_5
    m n : Nat
    β : Type u_6
    γ : Type u_7
    C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
    v : List.Vector α n
    w : List.Vector β n
    nil : C List.Vector.nil List.Vector.nil
    cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
    ⊢ C v w
  -/
  induction' n with n ih
    /-
      case zero
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      v : List.Vector α 0
      w : List.Vector β 0
      ⊢ C v w
    -/
  · rcases v with ⟨_ | ⟨-, -⟩, - | -⟩
    /-
      case zero.mk.nil.refl
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      w : List.Vector β 0
      ⊢ C ⟨List.nil, ⋯⟩ w
    -/
    rcases w with ⟨_ | ⟨-, -⟩, - | -⟩
    /-
      case zero.mk.nil.refl.mk.nil.refl
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      ⊢ C ⟨List.nil, ⋯⟩ ⟨List.nil, ⋯⟩
    -/
    exact nil
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      n : Nat
      ih : (v : List.Vector α n) → (w : List.Vector β n) → C v w
      v : List.Vector α (HAdd.hAdd n 1)
      w : List.Vector β (HAdd.hAdd n 1)
      ⊢ C v w
    -/
  · rcases v with ⟨_ | ⟨a, v⟩, v_property⟩
    /-
      case succ.mk.nil
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      n : Nat
      ih : (v : List.Vector α n) → (w : List.Vector β n) → C v w
      w : List.Vector β (HAdd.hAdd n 1)
      v_property : Eq List.nil.length (HAdd.hAdd n 1)
      ⊢ C ⟨List.nil, v_property⟩ w
    -/
    cases v_property
    /-
      case succ.mk.cons
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      n : Nat
      ih : (v : List.Vector α n) → (w : List.Vector β n) → C v w
      w : List.Vector β (HAdd.hAdd n 1)
      a : α
      v : List α
      v_property : Eq (List.cons a v).length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a v, v_property⟩ w
    -/
    rcases w with ⟨_ | ⟨b, w⟩, w_property⟩
    /-
      case succ.mk.cons.mk.nil
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      n : Nat
      ih : (v : List.Vector α n) → (w : List.Vector β n) → C v w
      a : α
      v : List α
      v_property : Eq (List.cons a v).length (HAdd.hAdd n 1)
      w_property : Eq List.nil.length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a v, v_property⟩ ⟨List.nil, w_property⟩
    -/
    cases w_property
    /-
      case succ.mk.cons.mk.cons
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      n : Nat
      ih : (v : List.Vector α n) → (w : List.Vector β n) → C v w
      a : α
      v : List α
      v_property : Eq (List.cons a v).length (HAdd.hAdd n 1)
      b : β
      w : List β
      w_property : Eq (List.cons b w).length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a v, v_property⟩ ⟨List.cons b w, w_property⟩
    -/
    apply @cons n _ _ ⟨v, (add_left_inj 1).mp v_property⟩ ⟨w, (add_left_inj 1).mp w_property⟩
    /-
      case succ.mk.cons.mk.cons
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {x : List.Vector α n} → {y : List.Vecto …
      n : Nat
      ih : (v : List.Vector α n) → (w : List.Vector β n) → C v w
      a : α
      v : List α
      v_property : Eq (List.cons a v).length (HAdd.hAdd n 1)
      b : β
      w : List β
      w_property : Eq (List.cons b w).length (HAdd.hAdd n 1)
      ⊢ C ⟨v, ⋯⟩ ⟨w, ⋯⟩
    -/
    apply ih
    /-
      🎉 no goals
    -/


/-- Define `C u v w` by induction on a triplet of vectors
`u : Vector α n`, `v : Vector β n`, and `w : Vector γ b`. -/
@[elab_as_elim]
def inductionOn₃ {C : ∀ {n}, Vector α n → Vector β n → Vector γ n → Sort*}
    (u : Vector α n) (v : Vector β n) (w : Vector γ n) (nil : C nil nil nil)
    (cons : ∀ {n a b c} {x : Vector α n} {y z}, C x y z → C (a ::ᵥ x) (b ::ᵥ y) (c ::ᵥ z)) :
    C u v w := by
  -- Porting note: removed `generalizing`: already generalized
  /-
    α : Type u_1
    β✝ : Type u_2
    γ✝ : Type u_3
    σ : Type u_4
    φ : Type u_5
    m n : Nat
    β : Type u_6
    γ : Type u_7
    C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
    u : List.Vector α n
    v : List.Vector β n
    w : List.Vector γ n
    nil : C List.Vector.nil List.Vector.nil List.Vector.nil
    cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
    ⊢ C u v w
  -/
  induction' n with n ih
    /-
      case zero
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      u : List.Vector α 0
      v : List.Vector β 0
      w : List.Vector γ 0
      ⊢ C u v w
    -/
  · rcases u with ⟨_ | ⟨-, -⟩, - | -⟩
    /-
      case zero.mk.nil.refl
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      v : List.Vector β 0
      w : List.Vector γ 0
      ⊢ C ⟨List.nil, ⋯⟩ v w
    -/
    rcases v with ⟨_ | ⟨-, -⟩, - | -⟩
    /-
      case zero.mk.nil.refl.mk.nil.refl
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      w : List.Vector γ 0
      ⊢ C ⟨List.nil, ⋯⟩ ⟨List.nil, ⋯⟩ w
    -/
    rcases w with ⟨_ | ⟨-, -⟩, - | -⟩
    /-
      case zero.mk.nil.refl.mk.nil.refl.mk.nil.refl
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      ⊢ C ⟨List.nil, ⋯⟩ ⟨List.nil, ⋯⟩ ⟨List.nil, ⋯⟩
    -/
    exact nil
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      u : List.Vector α (HAdd.hAdd n 1)
      v : List.Vector β (HAdd.hAdd n 1)
      w : List.Vector γ (HAdd.hAdd n 1)
      ⊢ C u v w
    -/
  · rcases u with ⟨_ | ⟨a, u⟩, u_property⟩
    /-
      case succ.mk.nil
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      v : List.Vector β (HAdd.hAdd n 1)
      w : List.Vector γ (HAdd.hAdd n 1)
      u_property : Eq List.nil.length (HAdd.hAdd n 1)
      ⊢ C ⟨List.nil, u_property⟩ v w
    -/
    cases u_property
    /-
      case succ.mk.cons
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      v : List.Vector β (HAdd.hAdd n 1)
      w : List.Vector γ (HAdd.hAdd n 1)
      a : α
      u : List α
      u_property : Eq (List.cons a u).length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a u, u_property⟩ v w
    -/
    rcases v with ⟨_ | ⟨b, v⟩, v_property⟩
    /-
      case succ.mk.cons.mk.nil
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      w : List.Vector γ (HAdd.hAdd n 1)
      a : α
      u : List α
      u_property : Eq (List.cons a u).length (HAdd.hAdd n 1)
      v_property : Eq List.nil.length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a u, u_property⟩ ⟨List.nil, v_property⟩ w
    -/
    cases v_property
    /-
      case succ.mk.cons.mk.cons
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      w : List.Vector γ (HAdd.hAdd n 1)
      a : α
      u : List α
      u_property : Eq (List.cons a u).length (HAdd.hAdd n 1)
      b : β
      v : List β
      v_property : Eq (List.cons b v).length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a u, u_property⟩ ⟨List.cons b v, v_property⟩ w
    -/
    rcases w with ⟨_ | ⟨c, w⟩, w_property⟩
    /-
      case succ.mk.cons.mk.cons.mk.nil
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      a : α
      u : List α
      u_property : Eq (List.cons a u).length (HAdd.hAdd n 1)
      b : β
      v : List β
      v_property : Eq (List.cons b v).length (HAdd.hAdd n 1)
      w_property : Eq List.nil.length (HAdd.hAdd n 1)
      ⊢ C ⟨List.cons a u, u_property⟩ ⟨List.cons b v, v_property⟩ ⟨List.nil, w_prope …
    -/
    cases w_property
    apply
      @cons n _ _ _ ⟨u, (add_left_inj 1).mp u_property⟩ ⟨v, (add_left_inj 1).mp v_property⟩
        ⟨w, (add_left_inj 1).mp w_property⟩
    /-
      case succ.mk.cons.mk.cons.mk.cons
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n✝ : Nat
      β : Type u_6
      γ : Type u_7
      C : {n : Nat} → List.Vector α n → List.Vector β n → List.Vector γ n → Sort u_8
      nil : C List.Vector.nil List.Vector.nil List.Vector.nil
      cons : {n : Nat} → {a : α} → {b : β} → {c : γ} → {x : List.Vector α n} → {y :  …
      n : Nat
      ih : (u : List.Vector α n) → (v : List.Vector β n) → (w : List.Vector γ n) → C …
      a : α
      u : List α
      u_property : Eq (List.cons a u).length (HAdd.hAdd n 1)
      b : β
      v : List β
      v_property : Eq (List.cons b v).length (HAdd.hAdd n 1)
      c : γ
      w : List γ
      w_property : Eq (List.cons c w).length (HAdd.hAdd n 1)
      ⊢ C ⟨u, ⋯⟩ ⟨v, ⋯⟩ ⟨w, ⋯⟩
    -/
    apply ih
    /-
      🎉 no goals
    -/


/-- Define `motive v` by case-analysis on `v : Vector α n`. -/
def casesOn {motive : ∀ {n}, Vector α n → Sort*} (v : Vector α m)
    (nil : motive nil)
    (cons : ∀ {n}, (hd : α) → (tl : Vector α n) → motive (Vector.cons hd tl)) :
    motive v :=
  inductionOn (C := motive) v nil @fun _ hd tl _ => cons hd tl


/-- Define `motive v₁ v₂` by case-analysis on `v₁ : Vector α n` and `v₂ : Vector β n`. -/
def casesOn₂  {motive : ∀{n}, Vector α n → Vector β n → Sort*} (v₁ : Vector α m) (v₂ : Vector β m)
    (nil : motive nil nil)
    (cons : ∀{n}, (x : α) → (y : β) → (xs : Vector α n) → (ys : Vector β n)
      → motive (x ::ᵥ xs) (y ::ᵥ ys)) :
    motive v₁ v₂ :=
  inductionOn₂ (C := motive) v₁ v₂ nil @fun _ x y xs ys _ => cons x y xs ys


/-- Define `motive v₁ v₂ v₃` by case-analysis on `v₁ : Vector α n`, `v₂ : Vector β n`, and
    `v₃ : Vector γ n`. -/
def casesOn₃  {motive : ∀{n}, Vector α n → Vector β n → Vector γ n → Sort*} (v₁ : Vector α m)
    (v₂ : Vector β m) (v₃ : Vector γ m) (nil : motive nil nil nil)
    (cons : ∀{n}, (x : α) → (y : β) → (z : γ) → (xs : Vector α n) → (ys : Vector β n)
      → (zs : Vector γ n) → motive (x ::ᵥ xs) (y ::ᵥ ys) (z ::ᵥ zs)) :
    motive v₁ v₂ v₃ :=
  inductionOn₃ (C := motive) v₁ v₂ v₃ nil @fun _ x y z xs ys zs _ => cons x y z xs ys zs


/-- Cast a vector to an array. -/
def toArray : Vector α n → Array α
                        /-
                          α : Type u_1
                          β✝ : Type u_2
                          γ✝ : Type u_3
                          σ : Type u_4
                          φ : Type u_5
                          m n : Nat
                          β : Type u_6
                          γ : Type u_7
                          xs : List α
                          property✝ : Eq xs.length n
                          ⊢ Eq (Array α) (Array α)
                        -/
  | ⟨xs, _⟩ => cast (by rfl) xs.toArray
                        /-
                          🎉 no goals
                        -/


/-- `v.insertIdx a i` inserts `a` into the vector `v` at position `i`
(and shifting later components to the right). -/
def insertIdx (a : α) (i : Fin (n + 1)) (v : Vector α n) : Vector α (n + 1) :=
  ⟨v.1.insertIdx i a, by
    /-
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      a✝ a : α
      i : Fin (HAdd.hAdd n 1)
      v : List.Vector α n
      ⊢ Eq (List.insertIdx (↑i) a ↑v).length (HAdd.hAdd n 1)
    -/
    rw [List.length_insertIdx, v.2]
    /-
      α : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      σ : Type u_4
      φ : Type u_5
      m n : Nat
      β : Type u_6
      γ : Type u_7
      a✝ a : α
      i : Fin (HAdd.hAdd n 1)
      v : List.Vector α n
      ⊢ Eq (ite (LE.le (↑i) n) (HAdd.hAdd n 1) n) (HAdd.hAdd n 1)
    -/
              /-
                🎉 no goals
              -/
    split <;> omega⟩
              /-
                🎉 no goals
              -/


@[deprecated (since := "2024-10-21")] alias insertNth := insertIdx


theorem insertIdx_val {i : Fin (n + 1)} {v : Vector α n} :
    (v.insertIdx a i).val = v.val.insertIdx i.1 a :=
  rfl


@[deprecated (since := "2024-10-21")] alias insertNth_val := insertIdx_val


@[simp]
theorem eraseIdx_val {i : Fin n} : ∀ {v : Vector α n}, (eraseIdx i v).val = v.val.eraseIdx i
  | _ => rfl


@[deprecated (since := "2024-10-21")] alias eraseNth_val := eraseIdx_val

@[deprecated (since := "2024-05-04")] alias removeNth_val := eraseIdx_val


theorem eraseIdx_insertIdx {v : Vector α n} {i : Fin (n + 1)} :
    eraseIdx i (insertIdx a i v) = v :=
  Subtype.eq <| List.eraseIdx_insertIdx i.1 v.1


@[deprecated (since := "2024-05-04")] alias eraseIdx_insertNth := eraseIdx_insertIdx

@[deprecated (since := "2024-05-04")] alias removeNth_insertNth := eraseIdx_insertIdx


/-- Erasing an element after inserting an element, at different indices. -/
theorem eraseIdx_insertIdx' {v : Vector α (n + 1)} :
    ∀ {i : Fin (n + 1)} {j : Fin (n + 2)},
      eraseIdx (j.succAbove i) (insertIdx a j v) = insertIdx a (i.predAbove j) (eraseIdx i v)
  | ⟨i, hi⟩, ⟨j, hj⟩ => by
    /-
      α : Type u_1
      n : Nat
      a : α
      v : List.Vector α (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i (HAdd.hAdd n 1)
      j : Nat
      hj : LT.lt j (HAdd.hAdd n 2)
      ⊢ Eq (List.Vector.eraseIdx (⟨j, hj⟩.succAbove ⟨i, hi⟩) (List.Vector.insertIdx  …
    -/
    dsimp [insertIdx, eraseIdx, Fin.succAbove, Fin.predAbove]
    /-
      α : Type u_1
      n : Nat
      a : α
      v : List.Vector α (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i (HAdd.hAdd n 1)
      j : Nat
      hj : LT.lt j (HAdd.hAdd n 2)
      ⊢ Eq ⟨(List.insertIdx j a ↑v).eraseIdx ↑(ite (LT.lt ⟨i, ⋯⟩ ⟨j, hj⟩) ⟨i, ⋯⟩ ⟨HA …
    -/
    rw [Subtype.mk_eq_mk]
    /-
      α : Type u_1
      n : Nat
      a : α
      v : List.Vector α (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i (HAdd.hAdd n 1)
      j : Nat
      hj : LT.lt j (HAdd.hAdd n 2)
      ⊢ Eq ((List.insertIdx j a ↑v).eraseIdx ↑(ite (LT.lt ⟨i, ⋯⟩ ⟨j, hj⟩) ⟨i, ⋯⟩ ⟨HA …
    -/
    simp only [Fin.lt_iff_val_lt_val]
    /-
      α : Type u_1
      n : Nat
      a : α
      v : List.Vector α (HAdd.hAdd n 1)
      i : Nat
      hi : LT.lt i (HAdd.hAdd n 1)
      j : Nat
      hj : LT.lt j (HAdd.hAdd n 2)
      ⊢ Eq ((List.insertIdx j a ↑v).eraseIdx ↑(ite (LT.lt i j) ⟨i, ⋯⟩ ⟨HAdd.hAdd i 1 …
    -/
    split_ifs with hij
    · rcases Nat.exists_eq_succ_of_ne_zero
        (Nat.pos_iff_ne_zero.1 (lt_of_le_of_lt (Nat.zero_le _) hij)) with ⟨j, rfl⟩
      /-
        case pos.intro
        α : Type u_1
        n : Nat
        a : α
        v : List.Vector α (HAdd.hAdd n 1)
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        j : Nat
        hj : LT.lt j.succ (HAdd.hAdd n 2)
        hij : LT.lt i j.succ
        ⊢ Eq ((List.insertIdx j.succ a ↑v).eraseIdx ↑⟨i, ⋯⟩) (List.insertIdx (↑(⟨j.suc …
      -/
      rw [← List.insertIdx_eraseIdx_of_ge]
        /-
          case pos.intro
          α : Type u_1
          n : Nat
          a : α
          v : List.Vector α (HAdd.hAdd n 1)
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          j : Nat
          hj : LT.lt j.succ (HAdd.hAdd n 2)
          hij : LT.lt i j.succ
          ⊢ Eq (List.insertIdx j a ((↑v).eraseIdx ↑⟨i, ⋯⟩)) (List.insertIdx (↑(⟨j.succ,  …
        -/
      · simp; rfl
              /-
                🎉 no goals
              -/
        /-
          case pos.intro.a
          α : Type u_1
          n : Nat
          a : α
          v : List.Vector α (HAdd.hAdd n 1)
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          j : Nat
          hj : LT.lt j.succ (HAdd.hAdd n 2)
          hij : LT.lt i j.succ
          ⊢ LT.lt (↑⟨i, ⋯⟩) (↑v).length
        -/
      · simpa
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.a
          α : Type u_1
          n : Nat
          a : α
          v : List.Vector α (HAdd.hAdd n 1)
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          j : Nat
          hj : LT.lt j.succ (HAdd.hAdd n 2)
          hij : LT.lt i j.succ
          ⊢ LE.le (↑⟨i, ⋯⟩) j
        -/
      · simpa [Nat.lt_succ_iff] using hij
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        n : Nat
        a : α
        v : List.Vector α (HAdd.hAdd n 1)
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        j : Nat
        hj : LT.lt j (HAdd.hAdd n 2)
        hij : Not (LT.lt i j)
        ⊢ Eq ((List.insertIdx j a ↑v).eraseIdx ↑⟨HAdd.hAdd i 1, ⋯⟩) (List.insertIdx (↑ …
      -/
    · dsimp
      /-
        case neg
        α : Type u_1
        n : Nat
        a : α
        v : List.Vector α (HAdd.hAdd n 1)
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        j : Nat
        hj : LT.lt j (HAdd.hAdd n 2)
        hij : Not (LT.lt i j)
        ⊢ Eq ((List.insertIdx j a ↑v).eraseIdx (HAdd.hAdd i 1)) (List.insertIdx j a ↑( …
      -/
      rw [← List.insertIdx_eraseIdx_of_le i j _ _ _]
        /-
          case neg
          α : Type u_1
          n : Nat
          a : α
          v : List.Vector α (HAdd.hAdd n 1)
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          j : Nat
          hj : LT.lt j (HAdd.hAdd n 2)
          hij : Not (LT.lt i j)
          ⊢ Eq (List.insertIdx j a ((↑v).eraseIdx i)) (List.insertIdx j a ↑(List.Vector. …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          n : Nat
          a : α
          v : List.Vector α (HAdd.hAdd n 1)
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          j : Nat
          hj : LT.lt j (HAdd.hAdd n 2)
          hij : Not (LT.lt i j)
          ⊢ LT.lt i (↑v).length
        -/
      · simpa
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          n : Nat
          a : α
          v : List.Vector α (HAdd.hAdd n 1)
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          j : Nat
          hj : LT.lt j (HAdd.hAdd n 2)
          hij : Not (LT.lt i j)
          ⊢ LE.le j i
        -/
      · simpa [not_lt] using hij
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-05-04")] alias eraseIdx_insertNth' := eraseIdx_insertIdx'

@[deprecated (since := "2024-05-04")] alias removeNth_insertNth' := eraseIdx_insertIdx'


theorem insertIdx_comm (a b : α) (i j : Fin (n + 1)) (h : i ≤ j) :
    ∀ v : Vector α n,
      (v.insertIdx a i).insertIdx b j.succ = (v.insertIdx b j).insertIdx a (Fin.castSucc i)
  | ⟨l, hl⟩ => by
    /-
      α : Type u_1
      n : Nat
      a b : α
      i j : Fin (HAdd.hAdd n 1)
      h : LE.le i j
      l : List α
      hl : Eq l.length n
      ⊢ Eq (List.Vector.insertIdx b j.succ (List.Vector.insertIdx a i ⟨l, hl⟩)) (Lis …
    -/
    refine Subtype.eq ?_
    /-
      α : Type u_1
      n : Nat
      a b : α
      i j : Fin (HAdd.hAdd n 1)
      h : LE.le i j
      l : List α
      hl : Eq l.length n
      ⊢ Eq ↑(List.Vector.insertIdx b j.succ (List.Vector.insertIdx a i ⟨l, hl⟩)) ↑(L …
    -/
    simp only [insertIdx_val, Fin.val_succ, Fin.castSucc, Fin.coe_castAdd]
    /-
      α : Type u_1
      n : Nat
      a b : α
      i j : Fin (HAdd.hAdd n 1)
      h : LE.le i j
      l : List α
      hl : Eq l.length n
      ⊢ Eq (List.insertIdx (HAdd.hAdd (↑j) 1) b (List.insertIdx (↑i) a l)) (List.ins …
    -/
    apply List.insertIdx_comm
      /-
        case x
        α : Type u_1
        n : Nat
        a b : α
        i j : Fin (HAdd.hAdd n 1)
        h : LE.le i j
        l : List α
        hl : Eq l.length n
        ⊢ LE.le ↑i ↑j
      -/
    · assumption
      /-
        🎉 no goals
      -/
      /-
        case x
        α : Type u_1
        n : Nat
        a b : α
        i j : Fin (HAdd.hAdd n 1)
        h : LE.le i j
        l : List α
        hl : Eq l.length n
        ⊢ LE.le (↑j) l.length
      -/
    · rw [hl]
      /-
        case x
        α : Type u_1
        n : Nat
        a b : α
        i j : Fin (HAdd.hAdd n 1)
        h : LE.le i j
        l : List α
        hl : Eq l.length n
        ⊢ LE.le (↑j) n
      -/
      exact Nat.le_of_succ_le_succ j.2
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-21")] alias insertNth_comm := insertIdx_comm


/-- `set v n a` replaces the `n`th element of `v` with `a`. -/
def set (v : Vector α n) (i : Fin n) (a : α) : Vector α n :=
                     /-
                       α : Type u_1
                       β✝ : Type u_2
                       γ✝ : Type u_3
                       σ : Type u_4
                       φ : Type u_5
                       m n : Nat
                       β : Type u_6
                       γ : Type u_7
                       v : List.Vector α n
                       i : Fin n
                       a : α
                       ⊢ Eq ((↑v).set (↑i) a).length n
                     -/
  ⟨v.1.set i.1 a, by simp⟩
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem toList_set (v : Vector α n) (i : Fin n) (a : α) :
    (v.set i a).toList = v.toList.set i a :=
  rfl


@[simp]
theorem get_set_same (v : Vector α n) (i : Fin n) (a : α) : (v.set i a).get i = a := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α n
    i : Fin n
    a : α
    ⊢ Eq ((v.set i a).get i) a
  -/
  cases v; cases i; simp [Vector.set, get_eq_get_toList]
                    /-
                      🎉 no goals
                    -/


theorem get_set_of_ne {v : Vector α n} {i j : Fin n} (h : i ≠ j) (a : α) :
    (v.set i a).get j = v.get j := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α n
    i j : Fin n
    h : Ne i j
    a : α
    ⊢ Eq ((v.set i a).get j) (v.get j)
  -/
  cases v; cases i; cases j
  /-
    case mk.mk.mk
    α : Type u_1
    n : Nat
    a : α
    val✝² : List α
    property✝ : Eq val✝².length n
    val✝¹ : Nat
    isLt✝¹ : LT.lt val✝¹ n
    val✝ : Nat
    isLt✝ : LT.lt val✝ n
    h : Ne ⟨val✝¹, isLt✝¹⟩ ⟨val✝, isLt✝⟩
    ⊢ Eq ((List.Vector.set ⟨val✝², property✝⟩ ⟨val✝¹, isLt✝¹⟩ a).get ⟨val✝, isLt✝⟩ …
  -/
  simp only [get_eq_get_toList, toList_set, toList_mk, Fin.cast_mk, List.get_eq_getElem]
  /-
    case mk.mk.mk
    α : Type u_1
    n : Nat
    a : α
    val✝² : List α
    property✝ : Eq val✝².length n
    val✝¹ : Nat
    isLt✝¹ : LT.lt val✝¹ n
    val✝ : Nat
    isLt✝ : LT.lt val✝ n
    h : Ne ⟨val✝¹, isLt✝¹⟩ ⟨val✝, isLt✝⟩
    ⊢ Eq (GetElem.getElem (val✝².set val✝¹ a) val✝ ⋯) (GetElem.getElem val✝² val✝ ⋯)
  -/
  rw [List.getElem_set_of_ne]
    /-
      case mk.mk.mk.h
      α : Type u_1
      n : Nat
      a : α
      val✝² : List α
      property✝ : Eq val✝².length n
      val✝¹ : Nat
      isLt✝¹ : LT.lt val✝¹ n
      val✝ : Nat
      isLt✝ : LT.lt val✝ n
      h : Ne ⟨val✝¹, isLt✝¹⟩ ⟨val✝, isLt✝⟩
      ⊢ Ne val✝¹ val✝
    -/
  · simpa using h
    /-
      🎉 no goals
    -/


theorem get_set_eq_if {v : Vector α n} {i j : Fin n} (a : α) :
    (v.set i a).get j = if i = j then a else v.get j := by
  /-
    α : Type u_1
    n : Nat
    v : List.Vector α n
    i j : Fin n
    a : α
    ⊢ Eq ((v.set i a).get j) (ite (Eq i j) a (v.get j))
  -/
                 /-
                   🎉 no goals
                 -/
  split_ifs <;> (try simp [*]); rwa [get_set_of_ne]
                                /-
                                  🎉 no goals
                                -/


@[to_additive]
theorem prod_set [Monoid α] (v : Vector α n) (i : Fin n) (a : α) :
    (v.set i a).toList.prod = (v.take i).toList.prod * a * (v.drop (i + 1)).toList.prod := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Monoid α
    v : List.Vector α n
    i : Fin n
    a : α
    ⊢ Eq (v.set i a).toList.prod (HMul.hMul (HMul.hMul (List.Vector.take (↑i) v).t …
  -/
  refine (List.prod_set v.toList i a).trans ?_
  /-
    α : Type u_1
    n : Nat
    inst✝ : Monoid α
    v : List.Vector α n
    i : Fin n
    a : α
    ⊢ Eq (HMul.hMul (HMul.hMul (List.take (↑i) v.toList).prod (ite (LT.lt (↑i) v.t …
  -/
  simp_all
  /-
    🎉 no goals
  -/


/-- Variant of `List.Vector.prod_set` that multiplies by the inverse of the replaced element.-/
@[to_additive
  "Variant of `List.Vector.sum_set` that subtracts the inverse of the replaced element."]
theorem prod_set' [CommGroup α] (v : Vector α n) (i : Fin n) (a : α) :
    (v.set i a).toList.prod = v.toList.prod * (v.get i)⁻¹ * a := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : CommGroup α
    v : List.Vector α n
    i : Fin n
    a : α
    ⊢ Eq (v.set i a).toList.prod (HMul.hMul (HMul.hMul v.toList.prod (Inv.inv (v.g …
  -/
  refine (List.prod_set' v.toList i a).trans ?_
  /-
    α : Type u_1
    n : Nat
    inst✝ : CommGroup α
    v : List.Vector α n
    i : Fin n
    a : α
    ⊢ Eq (HMul.hMul v.toList.prod (dite (LT.lt (↑i) v.toList.length) (fun hn => HM …
  -/
  simp [get_eq_get_toList, mul_assoc]
  /-
    🎉 no goals
  -/


private def traverseAux {α β : Type u} (f : α → F β) : ∀ x : List α, F (Vector β x.length)
  | [] => pure Vector.nil
  | x :: xs => Vector.cons <$> f x <*> traverseAux f xs


/-- Apply an applicative function to each component of a vector. -/
protected def traverse {α β : Type u} (f : α → F β) : Vector α n → F (Vector β n)
                        /-
                          α✝ : Type u_1
                          β✝ : Type u_2
                          γ : Type u_3
                          σ : Type u_4
                          φ : Type u_5
                          m n : Nat
                          F G : Type u → Type u
                          inst✝¹ : Applicative F
                          inst✝ : Applicative G
                          α β : Type u
                          f : α → F β
                          v : List α
                          Hv : Eq v.length n
                          ⊢ Eq (F (List.Vector β v.length)) (F (List.Vector β n))
                        -/
  | ⟨v, Hv⟩ => cast (by rw [Hv]) <| traverseAux f v
                        /-
                          🎉 no goals
                        -/


@[simp]
protected theorem traverse_def (f : α → F β) (x : α) :
    ∀ xs : Vector α n, (x ::ᵥ xs).traverse f = cons <$> f x <*> xs.traverse f := by
  /-
    n : Nat
    F : Type u → Type u
    inst✝ : Applicative F
    α β : Type u
    f : α → F β
    x : α
    ⊢ ∀ (xs : List.Vector α n), Eq (List.Vector.traverse f (List.Vector.cons x xs) …
  -/
  rintro ⟨xs, rfl⟩; rfl
                    /-
                      🎉 no goals
                    -/


protected theorem id_traverse : ∀ x : Vector α n, x.traverse (pure : _ → Id _) = x := by
  /-
    n : Nat
    α : Type u
    ⊢ ∀ (x : List.Vector α n), Eq (List.Vector.traverse Pure.pure x) x
  -/
  rintro ⟨x, rfl⟩; dsimp [Vector.traverse, cast]
  /-
    case mk
    α : Type u
    x : List α
    ⊢ Eq (List.Vector.traverseAux Pure.pure x) ⟨x, ⋯⟩
  -/
  induction' x with x xs IH; · rfl
                               /-
                                 🎉 no goals
                               -/
  /-
    case mk.cons
    α : Type u
    x : α
    xs : List α
    IH : Eq (List.Vector.traverseAux Pure.pure xs) ⟨xs, ⋯⟩
    ⊢ Eq (List.Vector.traverseAux Pure.pure (List.cons x xs)) ⟨List.cons x xs, ⋯⟩
  -/
  simp! [IH]; rfl
              /-
                🎉 no goals
              -/


@[nolint unusedArguments]
protected theorem comp_traverse (f : β → F γ) (g : α → G β) (x : Vector α n) :
    Vector.traverse (Comp.mk ∘ Functor.map f ∘ g) x =
      Comp.mk (Vector.traverse f <$> Vector.traverse g x) := by
  /-
    n : Nat
    F G : Type u → Type u
    inst✝² : Applicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    f : β → F γ
    g : α → G β
    x : List.Vector α n
    ⊢ Eq (List.Vector.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
  -/
  induction' x with n x xs ih
    /-
      case nil
      n : Nat
      F G : Type u → Type u
      inst✝² : Applicative F
      inst✝¹ : Applicative G
      inst✝ : LawfulApplicative G
      α β γ : Type u
      f : β → F γ
      g : α → G β
      ⊢ Eq (List.Vector.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
    -/
  · simp! [cast, *, functor_norm]
    /-
      case nil
      n : Nat
      F G : Type u → Type u
      inst✝² : Applicative F
      inst✝¹ : Applicative G
      inst✝ : LawfulApplicative G
      α β γ : Type u
      f : β → F γ
      g : α → G β
      ⊢ Eq (Pure.pure List.Vector.nil) (Functor.Comp.mk (Pure.pure (Pure.pure List.V …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      n✝ : Nat
      F G : Type u → Type u
      inst✝² : Applicative F
      inst✝¹ : Applicative G
      inst✝ : LawfulApplicative G
      α β γ : Type u
      f : β → F γ
      g : α → G β
      n : Nat
      x : α
      xs : List.Vector α n
      ih : Eq (List.Vector.traverse (Function.comp Functor.Comp.mk (Function.comp (F …
      ⊢ Eq (List.Vector.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
    -/
  · rw [Vector.traverse_def, ih]
    /-
      case cons
      n✝ : Nat
      F G : Type u → Type u
      inst✝² : Applicative F
      inst✝¹ : Applicative G
      inst✝ : LawfulApplicative G
      α β γ : Type u
      f : β → F γ
      g : α → G β
      n : Nat
      x : α
      xs : List.Vector α n
      ih : Eq (List.Vector.traverse (Function.comp Functor.Comp.mk (Function.comp (F …
      ⊢ Eq (Seq.seq (Functor.map List.Vector.cons (Function.comp Functor.Comp.mk (Fu …
    -/
    simp [functor_norm, Function.comp_def]
    /-
      🎉 no goals
    -/


protected theorem traverse_eq_map_id {α β} (f : α → β) :
    ∀ x : Vector α n, x.traverse ((pure : _ → Id _) ∘ f) = (pure : _ → Id _) (map f x) := by
  /-
    n : Nat
    α β : Type u_6
    f : α → β
    ⊢ ∀ (x : List.Vector α n), Eq (List.Vector.traverse (Function.comp Pure.pure f …
  -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  rintro ⟨x, rfl⟩; simp!; induction x <;> simp! [*, functor_norm] <;> rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


protected theorem naturality {α β : Type u} (f : α → F β) (x : Vector α n) :
    η (x.traverse f) = x.traverse (@η _ ∘ f) := by
  /-
    n : Nat
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative G
    inst✝ : LawfulApplicative F
    η : ApplicativeTransformation F G
    α β : Type u
    f : α → F β
    x : List.Vector α n
    ⊢ Eq ((fun {α} => η.app α) (List.Vector.traverse f x)) (List.Vector.traverse ( …
  -/
  induction' x with n x xs ih
    /-
      case nil
      n : Nat
      F G : Type u → Type u
      inst✝³ : Applicative F
      inst✝² : Applicative G
      inst✝¹ : LawfulApplicative G
      inst✝ : LawfulApplicative F
      η : ApplicativeTransformation F G
      α β : Type u
      f : α → F β
      ⊢ Eq ((fun {α} => η.app α) (List.Vector.traverse f List.Vector.nil)) (List.Vec …
    -/
  · simp! [functor_norm, cast, η.preserves_pure]
    /-
      🎉 no goals
    -/
    /-
      case cons
      n✝ : Nat
      F G : Type u → Type u
      inst✝³ : Applicative F
      inst✝² : Applicative G
      inst✝¹ : LawfulApplicative G
      inst✝ : LawfulApplicative F
      η : ApplicativeTransformation F G
      α β : Type u
      f : α → F β
      n : Nat
      x : α
      xs : List.Vector α n
      ih : Eq ((fun {α} => η.app α) (List.Vector.traverse f xs)) (List.Vector.traver …
      ⊢ Eq ((fun {α} => η.app α) (List.Vector.traverse f (List.Vector.cons x xs))) ( …
    -/
  · rw [Vector.traverse_def, Vector.traverse_def, ← ih, η.preserves_seq, η.preserves_map]
    /-
      case cons
      n✝ : Nat
      F G : Type u → Type u
      inst✝³ : Applicative F
      inst✝² : Applicative G
      inst✝¹ : LawfulApplicative G
      inst✝ : LawfulApplicative F
      η : ApplicativeTransformation F G
      α β : Type u
      f : α → F β
      n : Nat
      x : α
      xs : List.Vector α n
      ih : Eq ((fun {α} => η.app α) (List.Vector.traverse f xs)) (List.Vector.traver …
      ⊢ Eq (Seq.seq (Functor.map List.Vector.cons ((fun {α} => η.app α) (f x))) fun  …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance : Traversable.{u} (flip Vector n) where
  traverse := @Vector.traverse n
  map {α β} := @Vector.map.{u, u} α β n


instance : LawfulTraversable.{u} (flip Vector n) where
  id_traverse := @Vector.id_traverse n
  comp_traverse := Vector.comp_traverse
  traverse_eq_map_id := @Vector.traverse_eq_map_id n
  naturality := Vector.naturality
               /-
                 α : Type u_1
                 β : Type u_2
                 γ : Type u_3
                 σ : Type u_4
                 φ : Type u_5
                 m n : Nat
                 ⊢ ∀ {α : Type u} (x : flip List.Vector n α), Eq (Functor.map id x) x
               -/
  id_map := by intro _ x; cases x; simp! [(· <$> ·)]
                                   /-
                                     🎉 no goals
                                   -/
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   σ : Type u_4
                   φ : Type u_5
                   m n : Nat
                   ⊢ ∀ {α β γ : Type u} (g : α → β) (h : β → γ) (x : flip List.Vector n α), Eq (F …
                 -/
  comp_map := by intro _ _ _ _ _ x; cases x; simp! [(· <$> ·)]
                                             /-
                                               🎉 no goals
                                             -/
  map_const := rfl

-- Porting note: not porting meta instances
-- unsafe instance reflect [reflected_univ.{u}] {α : Type u} [has_reflect α]
--     [reflected _ α] {n : ℕ} : has_reflect (Vector α n) := fun v =>
--   @Vector.inductionOn α (fun n => reflected _) n v
--     ((by
--           trace
--             "./././Mathport/Syntax/Translate/Tactic/Builtin.lean:76:14:
--              unsupported tactic `reflect_name #[]" :
--           reflected _ @Vector.nil.{u}).subst
--       q(α))
--     fun n x xs ih =>
--     (by
--           trace
--             "./././Mathport/Syntax/Translate/Tactic/Builtin.lean:76:14:
--              unsupported tactic `reflect_name #[]" :
--           reflected _ @Vector.cons.{u}).subst₄
--       q(α) q(n) q(x) ih


@[simp]
theorem replicate_succ (val : α) :
    replicate (n+1) val = val ::ᵥ (replicate n val) :=
  rfl


                                                                       /-
                                                                         α : Type u_1
                                                                         β : Type u_2
                                                                         γ : Type u_3
                                                                         σ : Type u_4
                                                                         φ : Type u_5
                                                                         m n : Nat
                                                                         x : α
                                                                         y : β
                                                                         s : σ
                                                                         xs : List.Vector α n
                                                                         ys : List.Vector α m
                                                                         ⊢ LT.lt 0 (HAdd.hAdd n.succ m)
                                                                       -/
@[simp] lemma get_append_cons_zero : get (append (x ::ᵥ xs) ys) ⟨0, by omega⟩ = x := rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem get_append_cons_succ {i : Fin (n + m)} {h} :
    get (append (x ::ᵥ xs) ys) ⟨i+1, h⟩ = get (append xs ys) i :=
  rfl


@[simp]
theorem append_nil : append xs nil = xs := by
  /-
    α : Type u_1
    n : Nat
    xs : List.Vector α n
    ⊢ Eq (xs.append List.Vector.nil) xs
  -/
  cases xs; simp [append]
            /-
              🎉 no goals
            -/


@[simp]
theorem get_map₂ (v₁ : Vector α n) (v₂ : Vector β n) (f : α → β → γ) (i : Fin n) :
    get (map₂ f v₁ v₂) i = f (get v₁ i) (get v₂ i) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    n : Nat
    v₁ : List.Vector α n
    v₂ : List.Vector β n
    f : α → β → γ
    i : Fin n
    ⊢ Eq ((List.Vector.map₂ f v₁ v₂).get i) (f (v₁.get i) (v₂.get i))
  -/
  clear * - v₁ v₂
  induction v₁, v₂ using inductionOn₂ with
  | nil =>
    exact Fin.elim0 i
  | cons ih =>
    rw [map₂_cons]
    cases i using Fin.cases
    · simp only [get_zero, head_cons]
    · simp only [get_cons_succ, ih]


@[simp]
theorem mapAccumr_cons {f : α → σ → σ × β} :
    mapAccumr f (x ::ᵥ xs) s
    = let r := mapAccumr f xs s
      let q := f x r.1
      (q.1, q.2 ::ᵥ r.2) :=
  rfl


@[simp]
theorem mapAccumr₂_cons {f : α → β → σ → σ × φ} :
    mapAccumr₂ f (x ::ᵥ xs) (y ::ᵥ ys) s
    = let r := mapAccumr₂ f xs ys s
      let q := f x y r.1
      (q.1, q.2 ::ᵥ r.2) :=
  rfl


