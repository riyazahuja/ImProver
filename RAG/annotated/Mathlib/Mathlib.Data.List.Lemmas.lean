@[simp]
theorem length_flatMap (l : List α) (f : α → List β) :
    length (List.flatMap l f) = sum (map (length ∘ f) l) := by
  /-
    α : Type u_1
    β : Type u_2
    l : List α
    f : α → List β
    ⊢ Eq (l.flatMap f).length (List.map (Function.comp List.length f) l).sum
  -/
  rw [List.flatMap, length_flatten, map_map]
  /-
    🎉 no goals
  -/


lemma countP_flatMap (p : β → Bool) (l : List α) (f : α → List β) :
    countP p (l.flatMap f) = sum (map (countP p ∘ f) l) := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Bool
    l : List α
    f : α → List β
    ⊢ Eq (List.countP p (l.flatMap f)) (List.map (Function.comp (List.countP p) f) …
  -/
  rw [List.flatMap, countP_flatten, map_map]
  /-
    🎉 no goals
  -/


lemma count_flatMap [BEq β] (l : List α) (f : α → List β) (x : β) :
    count x (l.flatMap f) = sum (map (count x ∘ f) l) := countP_flatMap _ _ _


@[deprecated (since := "2024-08-20")] alias getElem_reverse' := getElem_reverse


@[deprecated (since := "2024-12-10")] alias tail_reverse_eq_reverse_dropLast := tail_reverse


@[deprecated (since := "2024-08-19")] alias nthLe_tail := getElem_tail


theorem injOn_insertIdx_index_of_not_mem (l : List α) (x : α) (hx : x ∉ l) :
    Set.InjOn (fun k => insertIdx k x l) { n | n ≤ l.length } := by
  /-
    α : Type u_1
    l : List α
    x : α
    hx : Not (Membership.mem l x)
    ⊢ Set.InjOn (fun k => List.insertIdx k x l) (setOf fun n => LE.le n l.length)
  -/
  induction' l with hd tl IH
    /-
      case nil
      α : Type u_1
      x : α
      hx : Not (Membership.mem List.nil x)
      ⊢ Set.InjOn (fun k => List.insertIdx k x List.nil) (setOf fun n => LE.le n Lis …
    -/
  · intro n hn m hm _
    /-
      case nil
      α : Type u_1
      x : α
      hx : Not (Membership.mem List.nil x)
      n : Nat
      hn : Membership.mem (setOf fun n => LE.le n List.nil.length) n
      m : Nat
      hm : Membership.mem (setOf fun n => LE.le n List.nil.length) m
      a✝ : Eq ((fun k => List.insertIdx k x List.nil) n) ((fun k => List.insertIdx k …
      ⊢ Eq n m
    -/
    simp_all [Set.mem_singleton_iff, Set.setOf_eq_eq_singleton, length]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      x hd : α
      tl : List α
      IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
      hx : Not (Membership.mem (List.cons hd tl) x)
      ⊢ Set.InjOn (fun k => List.insertIdx k x (List.cons hd tl)) (setOf fun n => LE …
    -/
  · intro n hn m hm h
    /-
      case cons
      α : Type u_1
      x hd : α
      tl : List α
      IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
      hx : Not (Membership.mem (List.cons hd tl) x)
      n : Nat
      hn : Membership.mem (setOf fun n => LE.le n (List.cons hd tl).length) n
      m : Nat
      hm : Membership.mem (setOf fun n => LE.le n (List.cons hd tl).length) m
      h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) n) ((fun k => List.ins …
      ⊢ Eq n m
    -/
    simp only [length, Set.mem_setOf_eq] at hn hm
    /-
      case cons
      α : Type u_1
      x hd : α
      tl : List α
      IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
      hx : Not (Membership.mem (List.cons hd tl) x)
      n : Nat
      hn : LE.le n (HAdd.hAdd tl.length 1)
      m : Nat
      hm : LE.le m (HAdd.hAdd tl.length 1)
      h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) n) ((fun k => List.ins …
      ⊢ Eq n m
    -/
    simp only [mem_cons, not_or] at hx
    /-
      case cons
      α : Type u_1
      x hd : α
      tl : List α
      IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
      n : Nat
      hn : LE.le n (HAdd.hAdd tl.length 1)
      m : Nat
      hm : LE.le m (HAdd.hAdd tl.length 1)
      h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) n) ((fun k => List.ins …
      hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
      ⊢ Eq n m
    -/
    cases n <;> cases m
      /-
        case cons.zero.zero
        α : Type u_1
        x hd : α
        tl : List α
        IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
        hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
        hn hm : LE.le 0 (HAdd.hAdd tl.length 1)
        h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) 0) ((fun k => List.ins …
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case cons.zero.succ
        α : Type u_1
        x hd : α
        tl : List α
        IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
        hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
        hn : LE.le 0 (HAdd.hAdd tl.length 1)
        n✝ : Nat
        hm : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
        h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) 0) ((fun k => List.ins …
        ⊢ Eq 0 (HAdd.hAdd n✝ 1)
      -/
    · simp [hx.left] at h
      /-
        🎉 no goals
      -/
      /-
        case cons.succ.zero
        α : Type u_1
        x hd : α
        tl : List α
        IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
        hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
        n✝ : Nat
        hn : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
        hm : LE.le 0 (HAdd.hAdd tl.length 1)
        h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) (HAdd.hAdd n✝ 1)) ((fu …
        ⊢ Eq (HAdd.hAdd n✝ 1) 0
      -/
    · simp [Ne.symm hx.left] at h
      /-
        🎉 no goals
      -/
      /-
        case cons.succ.succ
        α : Type u_1
        x hd : α
        tl : List α
        IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
        hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
        n✝¹ : Nat
        hn : LE.le (HAdd.hAdd n✝¹ 1) (HAdd.hAdd tl.length 1)
        n✝ : Nat
        hm : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
        h : Eq ((fun k => List.insertIdx k x (List.cons hd tl)) (HAdd.hAdd n✝¹ 1)) ((f …
        ⊢ Eq (HAdd.hAdd n✝¹ 1) (HAdd.hAdd n✝ 1)
      -/
    · simp only [true_and, eq_self_iff_true, insertIdx_succ_cons] at h
      /-
        case cons.succ.succ
        α : Type u_1
        x hd : α
        tl : List α
        IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
        hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
        n✝¹ : Nat
        hn : LE.le (HAdd.hAdd n✝¹ 1) (HAdd.hAdd tl.length 1)
        n✝ : Nat
        hm : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
        h : Eq (List.cons hd (List.insertIdx n✝¹ x tl)) (List.cons hd (List.insertIdx  …
        ⊢ Eq (HAdd.hAdd n✝¹ 1) (HAdd.hAdd n✝ 1)
      -/
      rw [Nat.succ_inj']
      /-
        case cons.succ.succ
        α : Type u_1
        x hd : α
        tl : List α
        IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
        hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
        n✝¹ : Nat
        hn : LE.le (HAdd.hAdd n✝¹ 1) (HAdd.hAdd tl.length 1)
        n✝ : Nat
        hm : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
        h : Eq (List.cons hd (List.insertIdx n✝¹ x tl)) (List.cons hd (List.insertIdx  …
        ⊢ Eq n✝¹ n✝
      -/
      refine IH hx.right ?_ ?_ (by injection h)
        /-
          case cons.succ.succ.refine_1
          α : Type u_1
          x hd : α
          tl : List α
          IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
          hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
          n✝¹ : Nat
          hn : LE.le (HAdd.hAdd n✝¹ 1) (HAdd.hAdd tl.length 1)
          n✝ : Nat
          hm : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
          h : Eq (List.cons hd (List.insertIdx n✝¹ x tl)) (List.cons hd (List.insertIdx  …
          ⊢ Membership.mem (setOf fun n => LE.le n tl.length) n✝¹
        -/
      · simpa [Nat.succ_le_succ_iff] using hn
        /-
          🎉 no goals
        -/
        /-
          case cons.succ.succ.refine_2
          α : Type u_1
          x hd : α
          tl : List α
          IH : Not (Membership.mem tl x) → Set.InjOn (fun k => List.insertIdx k x tl) (s …
          hx : And (Not (Eq x hd)) (Not (Membership.mem tl x))
          n✝¹ : Nat
          hn : LE.le (HAdd.hAdd n✝¹ 1) (HAdd.hAdd tl.length 1)
          n✝ : Nat
          hm : LE.le (HAdd.hAdd n✝ 1) (HAdd.hAdd tl.length 1)
          h : Eq (List.cons hd (List.insertIdx n✝¹ x tl)) (List.cons hd (List.insertIdx  …
          ⊢ Membership.mem (setOf fun n => LE.le n tl.length) n✝
        -/
      · simpa [Nat.succ_le_succ_iff] using hm
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-10-21")]
alias injOn_insertNth_index_of_not_mem := injOn_insertIdx_index_of_not_mem


theorem foldr_range_subset_of_range_subset {f : β → α → α} {g : γ → α → α}
    (hfg : Set.range f ⊆ Set.range g) (a : α) : Set.range (foldr f a) ⊆ Set.range (foldr g a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : β → α → α
    g : γ → α → α
    hfg : HasSubset.Subset (Set.range f) (Set.range g)
    a : α
    ⊢ HasSubset.Subset (Set.range (List.foldr f a)) (Set.range (List.foldr g a))
  -/
  rintro _ ⟨l, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : β → α → α
    g : γ → α → α
    hfg : HasSubset.Subset (Set.range f) (Set.range g)
    a : α
    l : List β
    ⊢ Membership.mem (Set.range (List.foldr g a)) (List.foldr f a l)
  -/
  induction' l with b l H
    /-
      case intro.nil
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : β → α → α
      g : γ → α → α
      hfg : HasSubset.Subset (Set.range f) (Set.range g)
      a : α
      ⊢ Membership.mem (Set.range (List.foldr g a)) (List.foldr f a List.nil)
    -/
  · exact ⟨[], rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : β → α → α
      g : γ → α → α
      hfg : HasSubset.Subset (Set.range f) (Set.range g)
      a : α
      b : β
      l : List β
      H : Membership.mem (Set.range (List.foldr g a)) (List.foldr f a l)
      ⊢ Membership.mem (Set.range (List.foldr g a)) (List.foldr f a (List.cons b l))
    -/
  · cases' hfg (Set.mem_range_self b) with c hgf
    /-
      case intro.cons.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : β → α → α
      g : γ → α → α
      hfg : HasSubset.Subset (Set.range f) (Set.range g)
      a : α
      b : β
      l : List β
      H : Membership.mem (Set.range (List.foldr g a)) (List.foldr f a l)
      c : γ
      hgf : Eq (g c) (f b)
      ⊢ Membership.mem (Set.range (List.foldr g a)) (List.foldr f a (List.cons b l))
    -/
    cases' H with m hgf'
    /-
      case intro.cons.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : β → α → α
      g : γ → α → α
      hfg : HasSubset.Subset (Set.range f) (Set.range g)
      a : α
      b : β
      l : List β
      c : γ
      hgf : Eq (g c) (f b)
      m : List γ
      hgf' : Eq (List.foldr g a m) (List.foldr f a l)
      ⊢ Membership.mem (Set.range (List.foldr g a)) (List.foldr f a (List.cons b l))
    -/
    rw [foldr_cons, ← hgf, ← hgf']
    /-
      case intro.cons.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : β → α → α
      g : γ → α → α
      hfg : HasSubset.Subset (Set.range f) (Set.range g)
      a : α
      b : β
      l : List β
      c : γ
      hgf : Eq (g c) (f b)
      m : List γ
      hgf' : Eq (List.foldr g a m) (List.foldr f a l)
      ⊢ Membership.mem (Set.range (List.foldr g a)) (g c (List.foldr g a m))
    -/
    exact ⟨c :: m, rfl⟩
    /-
      🎉 no goals
    -/


theorem foldl_range_subset_of_range_subset {f : α → β → α} {g : α → γ → α}
    (hfg : (Set.range fun a c => f c a) ⊆ Set.range fun b c => g c b) (a : α) :
    Set.range (foldl f a) ⊆ Set.range (foldl g a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → α
    g : α → γ → α
    hfg : HasSubset.Subset (Set.range fun a c => f c a) (Set.range fun b c => g c b)
    a : α
    ⊢ HasSubset.Subset (Set.range (List.foldl f a)) (Set.range (List.foldl g a))
  -/
  change (Set.range fun l => _) ⊆ Set.range fun l => _
  -- Porting note: This was simply `simp_rw [← foldr_reverse]`
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → α
    g : α → γ → α
    hfg : HasSubset.Subset (Set.range fun a c => f c a) (Set.range fun b c => g c b)
    a : α
    ⊢ HasSubset.Subset (Set.range fun l => List.foldl f a l) (Set.range fun l => L …
  -/
  simp_rw [← foldr_reverse _ (fun z w => g w z), ← foldr_reverse _ (fun z w => f w z)]
  -- Porting note: This `change` was not necessary in mathlib3
  change (Set.range (foldr (fun z w => f w z) a ∘ reverse)) ⊆
    Set.range (foldr (fun z w => g w z) a ∘ reverse)
  simp_rw [Set.range_comp _ reverse, reverse_involutive.bijective.surjective.range_eq,
    Set.image_univ]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → α
    g : α → γ → α
    hfg : HasSubset.Subset (Set.range fun a c => f c a) (Set.range fun b c => g c b)
    a : α
    ⊢ HasSubset.Subset (Set.range (List.foldr (fun z w => f w z) a)) (Set.range (L …
  -/
  exact foldr_range_subset_of_range_subset hfg a
  /-
    🎉 no goals
  -/


theorem foldr_range_eq_of_range_eq {f : β → α → α} {g : γ → α → α} (hfg : Set.range f = Set.range g)
    (a : α) : Set.range (foldr f a) = Set.range (foldr g a) :=
  (foldr_range_subset_of_range_subset hfg.le a).antisymm
    (foldr_range_subset_of_range_subset hfg.ge a)


theorem foldl_range_eq_of_range_eq {f : α → β → α} {g : α → γ → α}
    (hfg : (Set.range fun a c => f c a) = Set.range fun b c => g c b) (a : α) :
    Set.range (foldl f a) = Set.range (foldl g a) :=
  (foldl_range_subset_of_range_subset hfg.le a).antisymm
    (foldl_range_subset_of_range_subset hfg.ge a)




theorem mapAccumr_eq_foldr {σ : Type*} (f : α → σ → σ × β) : ∀ (as : List α) (s : σ),
    mapAccumr f as s = List.foldr (fun a s =>
                                    let r := f a s.1
                                    (r.1, r.2 :: s.2)
                                  ) (s, []) as
  | [], _ => rfl
  | a :: as, s => by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      f : α → σ → Prod σ β
      a : α
      as : List α
      s : σ
      ⊢ Eq (List.mapAccumr f (List.cons a as) s)
          (List.foldr
            (fun a s =>
              let r := f a s.1;
              { fst := r.1, snd := List.cons r.2 s.2 })
            { fst := s, snd := List.nil } (List.cons a as))
    -/
    simp only [mapAccumr, foldr, mapAccumr_eq_foldr f as]
    /-
      🎉 no goals
    -/


theorem mapAccumr₂_eq_foldr {σ φ : Type*} (f : α → β → σ → σ × φ) :
    ∀ (as : List α) (bs : List β) (s : σ),
    mapAccumr₂ f as bs s = foldr (fun ab s =>
                              let r := f ab.1 ab.2 s.1
                              (r.1, r.2 :: s.2)
                            ) (s, []) (as.zip bs)
  | [], [], _ => rfl
  | _ :: _, [], _ => rfl
  | [], _ :: _, _ => rfl
  | a :: as, b :: bs, s => by
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      φ : Type u_5
      f : α → β → σ → Prod σ φ
      a : α
      as : List α
      b : β
      bs : List β
      s : σ
      ⊢ Eq (List.mapAccumr₂ f (List.cons a as) (List.cons b bs) s)
          (List.foldr
            (fun ab s =>
              let r := f ab.1 ab.2 s.1;
              { fst := r.1, snd := List.cons r.2 s.2 })
            { fst := s, snd := List.nil } ((List.cons a as).zip (List.cons b bs)))
    -/
    simp only [mapAccumr₂, foldr, mapAccumr₂_eq_foldr f as]
    /-
      α : Type u_1
      β : Type u_2
      σ : Type u_4
      φ : Type u_5
      f : α → β → σ → Prod σ φ
      a : α
      as : List α
      b : β
      bs : List β
      s : σ
      ⊢ Eq { fst := (f a b (List.foldr (fun ab s => { fst := (f ab.1 ab.2 s.1).1, sn …
    -/
    rfl
    /-
      🎉 no goals
    -/


