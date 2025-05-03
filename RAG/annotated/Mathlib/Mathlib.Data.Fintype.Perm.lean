/-- Given a list, produce a list of all permutations of its elements. -/
def permsOfList : List α → List (Perm α)
  | [] => [1]
  | a :: l => permsOfList l ++ l.flatMap fun b => (permsOfList l).map fun f => Equiv.swap a b * f


theorem length_permsOfList : ∀ l : List α, length (permsOfList l) = l.length !
  | [] => rfl
  | a :: l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      ⊢ Eq (permsOfList (List.cons a l)).length (List.cons a l).length.factorial
    -/
    rw [length_cons, Nat.factorial_succ]
    simp only [permsOfList, length_append, length_permsOfList, length_flatMap, comp_def,
     length_map, map_const', sum_replicate, smul_eq_mul, succ_mul]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      ⊢ Eq (HAdd.hAdd l.length.factorial (HMul.hMul l.length l.length.factorial)) (H …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem mem_permsOfList_of_mem {l : List α} {f : Perm α} (h : ∀ x, f x ≠ x → x ∈ l) :
    f ∈ permsOfList l := by
  induction l generalizing f with
  | nil =>
    -- Porting note: applied `not_mem_nil` because it is no longer true definitionally.
    simp only [not_mem_nil] at h
    exact List.mem_singleton.2 (Equiv.ext fun x => Decidable.byContradiction <| h x)
  | cons a l IH =>
  by_cases hfa : f a = a
  · refine mem_append_left _ (IH fun x hx => mem_of_ne_of_mem ?_ (h x hx))
    rintro rfl
    exact hx hfa
  have hfa' : f (f a) ≠ f a := mt (fun h => f.injective h) hfa
  have : ∀ x : α, (Equiv.swap a (f a) * f) x ≠ x → x ∈ l := by
    intro x hx
    have hxa : x ≠ a := by
      rintro rfl
      apply hx
      simp only [mul_apply, swap_apply_right]
    refine List.mem_of_ne_of_mem hxa (h x fun h => ?_)
    simp only [mul_apply, swap_apply_def, mul_apply, Ne, apply_eq_iff_eq] at hx
    split_ifs at hx with h_1
    exacts [hxa (h.symm.trans h_1), hx h]
  suffices f ∈ permsOfList l ∨ ∃ b ∈ l, ∃ g ∈ permsOfList l, Equiv.swap a b * g = f by
    simpa only [permsOfList, exists_prop, List.mem_map, mem_append, List.mem_flatMap]
  refine or_iff_not_imp_left.2 fun _hfl => ⟨f a, ?_, Equiv.swap a (f a) * f, IH this, ?_⟩
  · exact mem_of_ne_of_mem hfa (h _ hfa')
  · rw [← mul_assoc, mul_def (swap a (f a)) (swap a (f a)), swap_swap, ← Perm.one_def, one_mul]


theorem mem_of_mem_permsOfList :
    -- Porting note: was `∀ {x}` but need to capture the `x`
    ∀ {l : List α} {f : Perm α}, f ∈ permsOfList l → (x : α ) → f x ≠ x → x ∈ l
  | [], f, h, heq_iff_eq => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      h : Membership.mem (permsOfList List.nil) f
      heq_iff_eq : α
      ⊢ Ne (f heq_iff_eq) heq_iff_eq → Membership.mem List.nil heq_iff_eq
    -/
    have : f = 1 := by simpa [permsOfList] using h
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      f : Equiv.Perm α
      h : Membership.mem (permsOfList List.nil) f
      heq_iff_eq : α
      this : Eq f 1
      ⊢ Ne (f heq_iff_eq) heq_iff_eq → Membership.mem List.nil heq_iff_eq
    -/
    rw [this]; simp
               /-
                 🎉 no goals
               -/
  | a :: l, f, h, x =>
    (mem_append.1 h).elim (fun h hx => mem_cons_of_mem _ (mem_of_mem_permsOfList h x hx))
      fun h hx =>
      let ⟨y, hy, hy'⟩ := List.mem_flatMap.1 h
      let ⟨g, hg₁, hg₂⟩ := List.mem_map.1 hy'
      -- Porting note: Seems like the implicit variable `x` of type `α` is needed.
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               a : α
                               l : List α
                               f : Equiv.Perm α
                               h✝ : Membership.mem (permsOfList (List.cons a l)) f
                               x : α
                               h : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.swa …
                               hx : Ne (f x) x
                               y : α
                               hy : Membership.mem l y
                               hy' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a y) f) (permsO …
                               g : Equiv.Perm α
                               hg₁ : Membership.mem (permsOfList l) g
                               hg₂ : Eq (HMul.hMul (Equiv.swap a y) g) f
                               hxa : Eq x a
                               ⊢ Membership.mem (List.cons a l) x
                             -/
      if hxa : x = a then by simp [hxa]
                             /-
                               🎉 no goals
                             -/
      else
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : DecidableEq α
                                                      a : α
                                                      l : List α
                                                      f : Equiv.Perm α
                                                      h✝ : Membership.mem (permsOfList (List.cons a l)) f
                                                      x : α
                                                      h : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.swa …
                                                      hx : Ne (f x) x
                                                      y : α
                                                      hy : Membership.mem l y
                                                      hy' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a y) f) (permsO …
                                                      g : Equiv.Perm α
                                                      hg₁ : Membership.mem (permsOfList l) g
                                                      hg₂ : Eq (HMul.hMul (Equiv.swap a y) g) f
                                                      hxa : Not (Eq x a)
                                                      hxy : Eq x y
                                                      ⊢ Membership.mem l x
                                                    -/
        if hxy : x = y then mem_cons_of_mem _ <| by rwa [hxy]
                                                    /-
                                                      🎉 no goals
                                                    -/
        else mem_cons_of_mem a <| mem_of_mem_permsOfList hg₁ _ <| by
              /-
                α : Type u_1
                inst✝ : DecidableEq α
                a : α
                l : List α
                f : Equiv.Perm α
                h✝ : Membership.mem (permsOfList (List.cons a l)) f
                x : α
                h : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.swa …
                hx : Ne (f x) x
                y : α
                hy : Membership.mem l y
                hy' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a y) f) (permsO …
                g : Equiv.Perm α
                hg₁ : Membership.mem (permsOfList l) g
                hg₂ : Eq (HMul.hMul (Equiv.swap a y) g) f
                hxa : Not (Eq x a)
                hxy : Not (Eq x y)
                ⊢ Ne (g x) x
              -/
              rw [eq_inv_mul_iff_mul_eq.2 hg₂, mul_apply, swap_inv, swap_apply_def]
              /-
                α : Type u_1
                inst✝ : DecidableEq α
                a : α
                l : List α
                f : Equiv.Perm α
                h✝ : Membership.mem (permsOfList (List.cons a l)) f
                x : α
                h : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.swa …
                hx : Ne (f x) x
                y : α
                hy : Membership.mem l y
                hy' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a y) f) (permsO …
                g : Equiv.Perm α
                hg₁ : Membership.mem (permsOfList l) g
                hg₂ : Eq (HMul.hMul (Equiv.swap a y) g) f
                hxa : Not (Eq x a)
                hxy : Not (Eq x y)
                ⊢ Ne (ite (Eq (f x) a) y (ite (Eq (f x) y) a (f x))) x
              -/
              split_ifs <;> [exact Ne.symm hxy; exact Ne.symm hxa; exact hx]
              /-
                🎉 no goals
              -/


theorem mem_permsOfList_iff {l : List α} {f : Perm α} :
    f ∈ permsOfList l ↔ ∀ {x}, f x ≠ x → x ∈ l :=
  ⟨mem_of_mem_permsOfList, mem_permsOfList_of_mem⟩


theorem nodup_permsOfList : ∀ {l : List α}, l.Nodup → (permsOfList l).Nodup
                /-
                  α : Type u_1
                  inst✝ : DecidableEq α
                  x✝ : List.nil.Nodup
                  ⊢ (permsOfList List.nil).Nodup
                -/
  | [], _ => by simp [permsOfList]
                /-
                  🎉 no goals
                -/
  | a :: l, hl => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      hl : (List.cons a l).Nodup
      ⊢ (permsOfList (List.cons a l)).Nodup
    -/
    have hl' : l.Nodup := hl.of_cons
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      hl : (List.cons a l).Nodup
      hl' : l.Nodup
      ⊢ (permsOfList (List.cons a l)).Nodup
    -/
    have hln' : (permsOfList l).Nodup := nodup_permsOfList hl'
    have hmeml : ∀ {f : Perm α}, f ∈ permsOfList l → f a = a := fun {f} hf =>
      not_not.1 (mt (mem_of_mem_permsOfList hf _) (nodup_cons.1 hl).1)
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      hl : (List.cons a l).Nodup
      hl' : l.Nodup
      hln' : (permsOfList l).Nodup
      hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
      ⊢ (permsOfList (List.cons a l)).Nodup
    -/
    rw [permsOfList, List.nodup_append, List.nodup_flatMap, pairwise_iff_getElem]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      l : List α
      hl : (List.cons a l).Nodup
      hl' : l.Nodup
      hln' : (permsOfList l).Nodup
      hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
      ⊢ And (permsOfList l).Nodup (And (And (∀ (x : α), Membership.mem l x → (List.m …
    -/
    refine ⟨?_, ⟨⟨?_,?_ ⟩, ?_⟩⟩
      /-
        case refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        ⊢ (permsOfList l).Nodup
      -/
    · exact hln'
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        ⊢ ∀ (x : α), Membership.mem l x → (List.map (fun f => HMul.hMul (Equiv.swap a  …
      -/
    · exact fun _ _ => hln'.map fun _ _ => mul_left_cancel
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        ⊢ ∀ (i j : Nat) (_hi : LT.lt i l.length) (_hj : LT.lt j l.length), LT.lt i j → …
      -/
    · intros i j hi hj hij x hx₁ hx₂
      /-
        case refine_3
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        i j : Nat
        hi : LT.lt i l.length
        hj : LT.lt j l.length
        hij : LT.lt i j
        x : Equiv.Perm α
        hx₁ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        hx₂ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        ⊢ False
      -/
      let ⟨f, hf⟩ := List.mem_map.1 hx₁
      /-
        case refine_3
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        i j : Nat
        hi : LT.lt i l.length
        hj : LT.lt j l.length
        hij : LT.lt i j
        x : Equiv.Perm α
        hx₁ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        hx₂ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        f : Equiv.Perm α
        hf : And (Membership.mem (permsOfList l) f) (Eq (HMul.hMul (Equiv.swap a (GetE …
        ⊢ False
      -/
      let ⟨g, hg⟩ := List.mem_map.1 hx₂
      have hix : x a = l[i] := by
        rw [← hf.2, mul_apply, hmeml hf.1, swap_apply_left]
      have hiy : x a = l[j] := by
        rw [← hg.2, mul_apply, hmeml hg.1, swap_apply_left]
      /-
        case refine_3
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        i j : Nat
        hi : LT.lt i l.length
        hj : LT.lt j l.length
        hij : LT.lt i j
        x : Equiv.Perm α
        hx₁ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        hx₂ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        f : Equiv.Perm α
        hf : And (Membership.mem (permsOfList l) f) (Eq (HMul.hMul (Equiv.swap a (GetE …
        g : Equiv.Perm α
        hg : And (Membership.mem (permsOfList l) g) (Eq (HMul.hMul (Equiv.swap a (GetE …
        hix : Eq (x a) (GetElem.getElem l i hi)
        hiy : Eq (x a) (GetElem.getElem l j hj)
        ⊢ False
      -/
      have hieqj : i = j := hl'.getElem_inj_iff.1 (hix.symm.trans hiy)
      /-
        case refine_3
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        i j : Nat
        hi : LT.lt i l.length
        hj : LT.lt j l.length
        hij : LT.lt i j
        x : Equiv.Perm α
        hx₁ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        hx₂ : Membership.mem ((fun b => List.map (fun f => HMul.hMul (Equiv.swap a b)  …
        f : Equiv.Perm α
        hf : And (Membership.mem (permsOfList l) f) (Eq (HMul.hMul (Equiv.swap a (GetE …
        g : Equiv.Perm α
        hg : And (Membership.mem (permsOfList l) g) (Eq (HMul.hMul (Equiv.swap a (GetE …
        hix : Eq (x a) (GetElem.getElem l i hi)
        hiy : Eq (x a) (GetElem.getElem l j hj)
        hieqj : Eq i j
        ⊢ False
      -/
      exact absurd hieqj (_root_.ne_of_lt hij)
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        ⊢ (permsOfList l).Disjoint (l.flatMap fun b => List.map (fun f => HMul.hMul (E …
      -/
    · intros f hf₁ hf₂
      /-
        case refine_4
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        f : Equiv.Perm α
        hf₁ : Membership.mem (permsOfList l) f
        hf₂ : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.s …
        ⊢ False
      -/
      let ⟨x, hx, hx'⟩ := List.mem_flatMap.1 hf₂
      /-
        case refine_4
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        f : Equiv.Perm α
        hf₁ : Membership.mem (permsOfList l) f
        hf₂ : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.s …
        x : α
        hx : Membership.mem l x
        hx' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a x) f) (permsO …
        ⊢ False
      -/
      let ⟨g, hg⟩ := List.mem_map.1 hx'
      /-
        case refine_4
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        f : Equiv.Perm α
        hf₁ : Membership.mem (permsOfList l) f
        hf₂ : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.s …
        x : α
        hx : Membership.mem l x
        hx' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a x) f) (permsO …
        g : Equiv.Perm α
        hg : And (Membership.mem (permsOfList l) g) (Eq (HMul.hMul (Equiv.swap a x) g) …
        ⊢ False
      -/
      have hgxa : g⁻¹ x = a := f.injective <| by rw [hmeml hf₁, ← hg.2]; simp
      /-
        case refine_4
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        l : List α
        hl : (List.cons a l).Nodup
        hl' : l.Nodup
        hln' : (permsOfList l).Nodup
        hmeml : ∀ {f : Equiv.Perm α}, Membership.mem (permsOfList l) f → Eq (f a) a
        f : Equiv.Perm α
        hf₁ : Membership.mem (permsOfList l) f
        hf₂ : Membership.mem (l.flatMap fun b => List.map (fun f => HMul.hMul (Equiv.s …
        x : α
        hx : Membership.mem l x
        hx' : Membership.mem (List.map (fun f => HMul.hMul (Equiv.swap a x) f) (permsO …
        g : Equiv.Perm α
        hg : And (Membership.mem (permsOfList l) g) (Eq (HMul.hMul (Equiv.swap a x) g) …
        hgxa : Eq ((Inv.inv g) x) a
        ⊢ False
      -/
      have hxa : x ≠ a := fun h => (List.nodup_cons.1 hl).1 (h ▸ hx)
      exact (List.nodup_cons.1 hl).1 <|
          hgxa ▸ mem_of_mem_permsOfList hg.1 _ (by rwa [apply_inv_self, hgxa])


/-- Given a finset, produce the finset of all permutations of its elements. -/
def permsOfFinset (s : Finset α) : Finset (Perm α) :=
  Quotient.hrecOn s.1 (fun l hl => ⟨permsOfList l, nodup_permsOfList hl⟩)
    (fun a b hab =>
      hfunext (congr_arg _ (Quotient.sound hab)) fun ha hb _ =>
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        γ : Type u_3
                                        inst✝¹ : DecidableEq α
                                        inst✝ : DecidableEq β
                                        s : Finset α
                                        a b : List α
                                        hab : HasEquiv.Equiv a b
                                        ha : Multiset.Nodup (Quotient.mk (List.isSetoid α) a)
                                        hb : Multiset.Nodup (Quotient.mk (List.isSetoid α) b)
                                        x✝ : HEq ha hb
                                        ⊢ ∀ (a_1 : Equiv.Perm α), Iff (Membership.mem { val := ↑(permsOfList a), nodup …
                                      -/
        heq_of_eq <| Finset.ext <| by simp [mem_permsOfList_iff, hab.mem_iff])
                                      /-
                                        🎉 no goals
                                      -/
    s.2


theorem mem_perms_of_finset_iff :
    ∀ {s : Finset α} {f : Perm α}, f ∈ permsOfFinset s ↔ ∀ {x}, f x ≠ x → x ∈ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ ∀ {s : Finset α} {f : Equiv.Perm α}, Iff (Membership.mem (permsOfFinset s) f …
  -/
  rintro ⟨⟨l⟩, hs⟩ f; exact mem_permsOfList_iff
                      /-
                        🎉 no goals
                      -/


theorem card_perms_of_finset : ∀ s : Finset α, #(permsOfFinset s) = (#s)! := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    ⊢ ∀ (s : Finset α), Eq (permsOfFinset s).card s.card.factorial
  -/
  rintro ⟨⟨l⟩, hs⟩; exact length_permsOfList l
                    /-
                      🎉 no goals
                    -/


/-- The collection of permutations of a fintype is a fintype. -/
def fintypePerm [Fintype α] : Fintype (Perm α) :=
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          γ : Type u_3
                                          inst✝² : DecidableEq α
                                          inst✝¹ : DecidableEq β
                                          inst✝ : Fintype α
                                          ⊢ ∀ (x : Equiv.Perm α), Membership.mem (permsOfFinset Finset.univ) x
                                        -/
  ⟨permsOfFinset (@Finset.univ α _), by simp [mem_perms_of_finset_iff]⟩
                                        /-
                                          🎉 no goals
                                        -/


instance Equiv.instFintype [Fintype α] [Fintype β] : Fintype (α ≃ β) :=
  if h : Fintype.card β = Fintype.card α then
    Trunc.recOnSubsingleton (Fintype.truncEquivFin α) fun eα =>
      Trunc.recOnSubsingleton (Fintype.truncEquivFin β) fun eβ =>
        @Fintype.ofEquiv _ (Perm α) fintypePerm
          (equivCongr (Equiv.refl α) (eα.trans (Eq.recOn h eβ.symm)) : α ≃ α ≃ (α ≃ β))
  else ⟨∅, fun x => False.elim (h (Fintype.card_eq.2 ⟨x.symm⟩))⟩


@[deprecated (since := "2024-11-19")] alias equivFintype := Equiv.instFintype


@[to_additive]
instance MulEquiv.instFintype
    {α β : Type*} [Mul α] [Mul β] [DecidableEq α] [DecidableEq β] [Fintype α] [Fintype β] :
    Fintype (α ≃* β) where
  elems := Equiv.instFintype.elems.filterMap
                                                                                            /-
                                                                                              α✝ : Type u_1
                                                                                              β✝ : Type u_2
                                                                                              γ : Type u_3
                                                                                              inst✝⁷ : DecidableEq α✝
                                                                                              inst✝⁶ : DecidableEq β✝
                                                                                              α : Type u_4
                                                                                              β : Type u_5
                                                                                              inst✝⁵ : Mul α
                                                                                              inst✝⁴ : Mul β
                                                                                              inst✝³ : DecidableEq α
                                                                                              inst✝² : DecidableEq β
                                                                                              inst✝¹ : Fintype α
                                                                                              inst✝ : Fintype β
                                                                                              ⊢ ∀ (a a' : Equiv α β) (b : MulEquiv α β), Membership.mem ((fun e => dite (∀ ( …
                                                                                            -/
    (fun e => if h : ∀ a b : α, e (a * b) = e a * e b then (⟨e, h⟩ : α ≃* β) else none) (by aesop)
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
                                                                                  /-
                                                                                    α✝ : Type u_1
                                                                                    β✝ : Type u_2
                                                                                    γ : Type u_3
                                                                                    inst✝⁷ : DecidableEq α✝
                                                                                    inst✝⁶ : DecidableEq β✝
                                                                                    α : Type u_4
                                                                                    β : Type u_5
                                                                                    inst✝⁵ : Mul α
                                                                                    inst✝⁴ : Mul β
                                                                                    inst✝³ : DecidableEq α
                                                                                    inst✝² : DecidableEq β
                                                                                    inst✝¹ : Fintype α
                                                                                    inst✝ : Fintype β
                                                                                    me : MulEquiv α β
                                                                                    ⊢ Eq (dite (∀ (a b : α), Eq (me.toEquiv (HMul.hMul a b)) (HMul.hMul (me.toEqui …
                                                                                  -/
  complete me := (Finset.mem_filterMap ..).mpr ⟨me.toEquiv, Finset.mem_univ _, by {simp; rfl}⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem Fintype.card_perm [Fintype α] : Fintype.card (Perm α) = (Fintype.card α)! :=
  Subsingleton.elim (@fintypePerm α _ _) (@Equiv.instFintype α α _ _ _ _) ▸ card_perms_of_finset _


theorem Fintype.card_equiv [Fintype α] [Fintype β] (e : α ≃ β) :
    Fintype.card (α ≃ β) = (Fintype.card α)! :=
  Fintype.card_congr (equivCongr (Equiv.refl α) e) ▸ Fintype.card_perm

