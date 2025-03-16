@[simp]
theorem zip_swap : ∀ (l₁ : List α) (l₂ : List β), (zip l₁ l₂).map Prod.swap = zip l₂ l₁
  | [], _ => zip_nil_right.symm
                 /-
                   α : Type u
                   β : Type u_1
                   l₁ : List α
                   ⊢ Eq (List.map Prod.swap (l₁.zip List.nil)) (List.nil.zip l₁)
                 -/
  | l₁, [] => by rw [zip_nil_right]; rfl
                                     /-
                                       🎉 no goals
                                     -/
  | a :: l₁, b :: l₂ => by
    /-
      α : Type u
      β : Type u_1
      a : α
      l₁ : List α
      b : β
      l₂ : List β
      ⊢ Eq (List.map Prod.swap ((List.cons a l₁).zip (List.cons b l₂))) ((List.cons  …
    -/
    simp only [zip_cons_cons, map_cons, zip_swap l₁ l₂, Prod.swap_prod_mk]
    /-
      🎉 no goals
    -/


theorem forall_zipWith {f : α → β → γ} {p : γ → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, length l₁ = length l₂ →
      (Forall p (zipWith f l₁ l₂) ↔ Forall₂ (fun x y => p (f x y)) l₁ l₂)
                    /-
                      α : Type u
                      β : Type u_1
                      γ : Type u_2
                      f : α → β → γ
                      p : γ → Prop
                      x✝ : Eq List.nil.length List.nil.length
                      ⊢ Iff (List.Forall p (List.zipWith f List.nil List.nil)) (List.Forall₂ (fun x  …
                    -/
  | [], [], _ => by simp
                    /-
                      🎉 no goals
                    -/
  | a :: l₁, b :: l₂, h => by
    /-
      α : Type u
      β : Type u_1
      γ : Type u_2
      f : α → β → γ
      p : γ → Prop
      a : α
      l₁ : List α
      b : β
      l₂ : List β
      h : Eq (List.cons a l₁).length (List.cons b l₂).length
      ⊢ Iff (List.Forall p (List.zipWith f (List.cons a l₁) (List.cons b l₂))) (List …
    -/
    simp only [length_cons, succ_inj'] at h
    /-
      α : Type u
      β : Type u_1
      γ : Type u_2
      f : α → β → γ
      p : γ → Prop
      a : α
      l₁ : List α
      b : β
      l₂ : List β
      h : Eq l₁.length l₂.length
      ⊢ Iff (List.Forall p (List.zipWith f (List.cons a l₁) (List.cons b l₂))) (List …
    -/
    simp [forall_zipWith h]
    /-
      🎉 no goals
    -/


theorem unzip_swap (l : List (α × β)) : unzip (l.map Prod.swap) = (unzip l).swap := by
  /-
    α : Type u
    β : Type u_1
    l : List (Prod α β)
    ⊢ Eq (List.map Prod.swap l).unzip l.unzip.swap
  -/
  simp only [unzip_eq_map, map_map]
  /-
    α : Type u
    β : Type u_1
    l : List (Prod α β)
    ⊢ Eq { fst := List.map (Function.comp Prod.fst Prod.swap) l, snd := List.map ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[congr]
theorem zipWith_congr (f g : α → β → γ) (la : List α) (lb : List β)
    (h : List.Forall₂ (fun a b => f a b = g a b) la lb) : zipWith f la lb = zipWith g la lb := by
  /-
    α : Type u
    β : Type u_1
    γ : Type u_2
    f g : α → β → γ
    la : List α
    lb : List β
    h : List.Forall₂ (fun a b => Eq (f a b) (g a b)) la lb
    ⊢ Eq (List.zipWith f la lb) (List.zipWith g la lb)
  -/
  induction' h with a b as bs hfg _ ih
    /-
      case nil
      α : Type u
      β : Type u_1
      γ : Type u_2
      f g : α → β → γ
      la : List α
      lb : List β
      ⊢ Eq (List.zipWith f List.nil List.nil) (List.zipWith g List.nil List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      β : Type u_1
      γ : Type u_2
      f g : α → β → γ
      la : List α
      lb : List β
      a : α
      b : β
      as : List α
      bs : List β
      hfg : Eq (f a b) (g a b)
      a✝ : List.Forall₂ (fun a b => Eq (f a b) (g a b)) as bs
      ih : Eq (List.zipWith f as bs) (List.zipWith g as bs)
      ⊢ Eq (List.zipWith f (List.cons a as) (List.cons b bs)) (List.zipWith g (List. …
    -/
  · exact congr_arg₂ _ hfg ih
    /-
      🎉 no goals
    -/


theorem zipWith_zipWith_left (f : δ → γ → ε) (g : α → β → δ) :
    ∀ (la : List α) (lb : List β) (lc : List γ),
      zipWith f (zipWith g la lb) lc = zipWith3 (fun a b c => f (g a b) c) la lb lc
  | [], _, _ => rfl
  | _ :: _, [], _ => rfl
  | _ :: _, _ :: _, [] => rfl
  | _ :: as, _ :: bs, _ :: cs => congr_arg (cons _) <| zipWith_zipWith_left f g as bs cs


theorem zipWith_zipWith_right (f : α → δ → ε) (g : β → γ → δ) :
    ∀ (la : List α) (lb : List β) (lc : List γ),
      zipWith f la (zipWith g lb lc) = zipWith3 (fun a b c => f a (g b c)) la lb lc
  | [], _, _ => rfl
  | _ :: _, [], _ => rfl
  | _ :: _, _ :: _, [] => rfl
  | _ :: as, _ :: bs, _ :: cs => congr_arg (cons _) <| zipWith_zipWith_right f g as bs cs


@[simp]
theorem zipWith3_same_left (f : α → α → β → γ) :
    ∀ (la : List α) (lb : List β), zipWith3 f la la lb = zipWith (fun a b => f a a b) la lb
  | [], _ => rfl
  | _ :: _, [] => rfl
  | _ :: as, _ :: bs => congr_arg (cons _) <| zipWith3_same_left f as bs


@[simp]
theorem zipWith3_same_mid (f : α → β → α → γ) :
    ∀ (la : List α) (lb : List β), zipWith3 f la lb la = zipWith (fun a b => f a b a) la lb
  | [], _ => rfl
  | _ :: _, [] => rfl
  | _ :: as, _ :: bs => congr_arg (cons _) <| zipWith3_same_mid f as bs


@[simp]
theorem zipWith3_same_right (f : α → β → β → γ) :
    ∀ (la : List α) (lb : List β), zipWith3 f la lb lb = zipWith (fun a b => f a b b) la lb
  | [], _ => rfl
  | _ :: _, [] => rfl
  | _ :: as, _ :: bs => congr_arg (cons _) <| zipWith3_same_right f as bs


instance (f : α → α → β) [IsSymmOp f] : IsSymmOp (zipWith f) :=
  ⟨zipWith_comm_of_comm f IsSymmOp.symm_op⟩


@[simp]
theorem length_revzip (l : List α) : length (revzip l) = length l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.revzip.length l.length
  -/
  simp only [revzip, length_zip, length_reverse, min_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem unzip_revzip (l : List α) : (revzip l).unzip = (l, l.reverse) :=
  unzip_zip (length_reverse l).symm


@[simp]
theorem revzip_map_fst (l : List α) : (revzip l).map Prod.fst = l := by
  /-
    α : Type u
    l : List α
    ⊢ Eq (List.map Prod.fst l.revzip) l
  -/
  rw [← unzip_fst, unzip_revzip]
  /-
    🎉 no goals
  -/


@[simp]
theorem revzip_map_snd (l : List α) : (revzip l).map Prod.snd = l.reverse := by
  /-
    α : Type u
    l : List α
    ⊢ Eq (List.map Prod.snd l.revzip) l.reverse
  -/
  rw [← unzip_snd, unzip_revzip]
  /-
    🎉 no goals
  -/


theorem reverse_revzip (l : List α) : reverse l.revzip = revzip l.reverse := by
  /-
    α : Type u
    l : List α
    ⊢ Eq l.revzip.reverse l.reverse.revzip
  -/
  rw [← zip_unzip (revzip l).reverse]
  /-
    α : Type u
    l : List α
    ⊢ Eq (l.revzip.reverse.unzip.1.zip l.revzip.reverse.unzip.2) l.reverse.revzip
  -/
  simp [unzip_eq_map, revzip, map_reverse, map_fst_zip, map_snd_zip]
  /-
    🎉 no goals
  -/


                                                                                     /-
                                                                                       α : Type u
                                                                                       l : List α
                                                                                       ⊢ Eq (List.map Prod.swap l.revzip) l.reverse.revzip
                                                                                     -/
theorem revzip_swap (l : List α) : (revzip l).map Prod.swap = revzip l.reverse := by simp [revzip]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[deprecated (since := "2024-07-29")] alias getElem?_zip_with := getElem?_zipWith'


theorem get?_zipWith' (f : α → β → γ) (l₁ : List α) (l₂ : List β) (i : ℕ) :
    (zipWith f l₁ l₂).get? i = ((l₁.get? i).map f).bind fun g => (l₂.get? i).map g := by
  /-
    α : Type u
    β : Type u_1
    γ : Type u_2
    f : α → β → γ
    l₁ : List α
    l₂ : List β
    i : Nat
    ⊢ Eq ((List.zipWith f l₁ l₂).get? i) ((Option.map f (l₁.get? i)).bind fun g => …
  -/
  simp [getElem?_zipWith']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-29")] alias get?_zip_with := get?_zipWith'

@[deprecated (since := "2024-07-29")] alias getElem?_zip_with_eq_some := getElem?_zipWith_eq_some


theorem get?_zipWith_eq_some (f : α → β → γ) (l₁ : List α) (l₂ : List β) (z : γ) (i : ℕ) :
    (zipWith f l₁ l₂).get? i = some z ↔
      ∃ x y, l₁.get? i = some x ∧ l₂.get? i = some y ∧ f x y = z := by
  /-
    α : Type u
    β : Type u_1
    γ : Type u_2
    f : α → β → γ
    l₁ : List α
    l₂ : List β
    z : γ
    i : Nat
    ⊢ Iff (Eq ((List.zipWith f l₁ l₂).get? i) (Option.some z)) (Exists fun x => Ex …
  -/
  simp [getElem?_zipWith_eq_some]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-29")] alias get?_zip_with_eq_some := get?_zipWith_eq_some


theorem get?_zip_eq_some (l₁ : List α) (l₂ : List β) (z : α × β) (i : ℕ) :
    (zip l₁ l₂).get? i = some z ↔ l₁.get? i = some z.1 ∧ l₂.get? i = some z.2 := by
  /-
    α : Type u
    β : Type u_1
    l₁ : List α
    l₂ : List β
    z : Prod α β
    i : Nat
    ⊢ Iff (Eq ((l₁.zip l₂).get? i) (Option.some z)) (And (Eq (l₁.get? i) (Option.s …
  -/
  simp [getElem?_zip_eq_some]
  /-
    🎉 no goals
  -/


@[deprecated getElem_zipWith (since := "2024-06-12")]
theorem get_zipWith {f : α → β → γ} {l : List α} {l' : List β} {i : Fin (zipWith f l l').length} :
    (zipWith f l l').get i =
      f (l.get ⟨i, lt_length_left_of_zipWith i.isLt⟩)
        (l'.get ⟨i, lt_length_right_of_zipWith i.isLt⟩) := by
  /-
    α : Type u
    β : Type u_1
    γ : Type u_2
    f : α → β → γ
    l : List α
    l' : List β
    i : Fin (List.zipWith f l l').length
    ⊢ Eq ((List.zipWith f l l').get i) (f (l.get ⟨↑i, ⋯⟩) (l'.get ⟨↑i, ⋯⟩))
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated getElem_zip (since := "2024-06-12")]
theorem get_zip {l : List α} {l' : List β} {i : Fin (zip l l').length} :
    (zip l l').get i =
      (l.get ⟨i, lt_length_left_of_zip i.isLt⟩, l'.get ⟨i, lt_length_right_of_zip i.isLt⟩) := by
  /-
    α : Type u
    β : Type u_1
    l : List α
    l' : List β
    i : Fin (l.zip l').length
    ⊢ Eq ((l.zip l').get i) { fst := l.get ⟨↑i, ⋯⟩, snd := l'.get ⟨↑i, ⋯⟩ }
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_zip_inits_tails {l : List α} {init tail : List α} :
    (init, tail) ∈ zip l.inits l.tails ↔ init ++ tail = l := by
  /-
    α : Type u
    l init tail : List α
    ⊢ Iff (Membership.mem (l.inits.zip l.tails) { fst := init, snd := tail }) (Eq  …
  -/
  induction' l with hd tl ih generalizing init tail <;> simp_rw [tails, inits, zip_cons_cons]
    /-
      case nil
      α : Type u
      init tail : List α
      ⊢ Iff (Membership.mem (List.cons { fst := List.nil, snd := List.nil } (List.ni …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u
      hd : α
      tl : List α
      ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
      init tail : List α
      ⊢ Iff (Membership.mem (List.cons { fst := List.nil, snd := List.cons hd tl } ( …
    -/
  · constructor <;> rw [mem_cons, zip_map_left, mem_map, Prod.exists]
      /-
        case cons.mp
        α : Type u
        hd : α
        tl : List α
        ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
        init tail : List α
        ⊢ Or (Eq { fst := init, snd := tail } { fst := List.nil, snd := List.cons hd t …
      -/
    · rintro (⟨rfl, rfl⟩ | ⟨_, _, h, rfl, rfl⟩)
        /-
          case cons.mp.inl.refl
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          ⊢ Eq (HAppend.hAppend List.nil (List.cons hd tl)) (List.cons hd tl)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case cons.mp.inr.intro.intro.intro.refl
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          w✝¹ w✝ : List α
          h : Membership.mem (tl.inits.zip tl.tails) { fst := w✝¹, snd := w✝ }
          ⊢ Eq (HAppend.hAppend ((fun t => List.cons hd t) w✝¹) (id w✝)) (List.cons hd tl)
        -/
      · simp [ih.mp h]
        /-
          🎉 no goals
        -/
      /-
        case cons.mpr
        α : Type u
        hd : α
        tl : List α
        ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
        init tail : List α
        ⊢ Eq (HAppend.hAppend init tail) (List.cons hd tl) → Or (Eq { fst := init, snd …
      -/
    · cases' init with hd' tl'
        /-
          case cons.mpr.nil
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          tail : List α
          ⊢ Eq (HAppend.hAppend List.nil tail) (List.cons hd tl) → Or (Eq { fst := List. …
        -/
      · rintro rfl
        /-
          case cons.mpr.nil
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          ⊢ Or (Eq { fst := List.nil, snd := List.cons hd tl } { fst := List.nil, snd := …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case cons.mpr.cons
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          tail : List α
          hd' : α
          tl' : List α
          ⊢ Eq (HAppend.hAppend (List.cons hd' tl') tail) (List.cons hd tl) → Or (Eq { f …
        -/
      · intro h
        /-
          case cons.mpr.cons
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          tail : List α
          hd' : α
          tl' : List α
          h : Eq (HAppend.hAppend (List.cons hd' tl') tail) (List.cons hd tl)
          ⊢ Or (Eq { fst := List.cons hd' tl', snd := tail } { fst := List.nil, snd := L …
        -/
        right
        /-
          case cons.mpr.cons.h
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          tail : List α
          hd' : α
          tl' : List α
          h : Eq (HAppend.hAppend (List.cons hd' tl') tail) (List.cons hd tl)
          ⊢ Exists fun a => Exists fun b => And (Membership.mem (tl.inits.zip tl.tails)  …
        -/
        use tl', tail
        /-
          case h
          α : Type u
          hd : α
          tl : List α
          ih : ∀ {init tail : List α}, Iff (Membership.mem (tl.inits.zip tl.tails) { fst …
          tail : List α
          hd' : α
          tl' : List α
          h : Eq (HAppend.hAppend (List.cons hd' tl') tail) (List.cons hd tl)
          ⊢ And (Membership.mem (tl.inits.zip tl.tails) { fst := tl', snd := tail }) (Eq …
        -/
        simp_all
        /-
          🎉 no goals
        -/


