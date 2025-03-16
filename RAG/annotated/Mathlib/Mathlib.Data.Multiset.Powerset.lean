/-- A helper function for the powerset of a multiset. Given a list `l`, returns a list
of sublists of `l` as multisets. -/
def powersetAux (l : List α) : List (Multiset α) :=
  (sublists l).map (↑)


theorem powersetAux_eq_map_coe {l : List α} : powersetAux l = (sublists l).map (↑) :=
  rfl


@[simp]
theorem mem_powersetAux {l : List α} {s} : s ∈ powersetAux l ↔ s ≤ ↑l :=
                               /-
                                 α : Type u_1
                                 l : List α
                                 s : Multiset α
                                 ⊢ ∀ (a : List α), Iff (Membership.mem (Multiset.powersetAux l) (Quotient.mk (L …
                               -/
  Quotient.inductionOn s <| by simp [powersetAux_eq_map_coe, Subperm, and_comm]
                               /-
                                 🎉 no goals
                               -/


/-- Helper function for the powerset of a multiset. Given a list `l`, returns a list
of sublists of `l` (using `sublists'`), as multisets. -/
def powersetAux' (l : List α) : List (Multiset α) :=
  (sublists' l).map (↑)


theorem powersetAux_perm_powersetAux' {l : List α} : powersetAux l ~ powersetAux' l := by
  /-
    α : Type u_1
    l : List α
    ⊢ (Multiset.powersetAux l).Perm (Multiset.powersetAux' l)
  -/
  rw [powersetAux_eq_map_coe]; exact (sublists_perm_sublists' _).map _
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem powersetAux'_nil : powersetAux' (@nil α) = [0] :=
  rfl


@[simp]
theorem powersetAux'_cons (a : α) (l : List α) :
    powersetAux' (a :: l) = powersetAux' l ++ List.map (cons a) (powersetAux' l) := by
  /-
    α : Type u_1
    a : α
    l : List α
    ⊢ Eq (Multiset.powersetAux' (List.cons a l)) (HAppend.hAppend (Multiset.powers …
  -/
  simp [powersetAux']
  /-
    🎉 no goals
  -/


theorem powerset_aux'_perm {l₁ l₂ : List α} (p : l₁ ~ l₂) : powersetAux' l₁ ~ powersetAux' l₂ := by
  induction p with
  | nil => simp
  | cons _ _ IH =>
    simp only [powersetAux'_cons]
    exact IH.append (IH.map _)
  | swap a b =>
    simp only [powersetAux'_cons, map_append, List.map_map, append_assoc]
    apply Perm.append_left
    rw [← append_assoc, ← append_assoc,
      (by funext s; simp [cons_swap] : cons b ∘ cons a = cons a ∘ cons b)]
    exact perm_append_comm.append_right _
  | trans _ _ IH₁ IH₂ => exact IH₁.trans IH₂


theorem powersetAux_perm {l₁ l₂ : List α} (p : l₁ ~ l₂) : powersetAux l₁ ~ powersetAux l₂ :=
  powersetAux_perm_powersetAux'.trans <|
    (powerset_aux'_perm p).trans powersetAux_perm_powersetAux'.symm

--Porting note (https://github.com/leanprover-community/mathlib4/issues/11083): slightly slower implementation due to `map ofList`

/-- The power set of a multiset. -/
def powerset (s : Multiset α) : Multiset (Multiset α) :=
  Quot.liftOn s
    (fun l => (powersetAux l : Multiset (Multiset α)))
    (fun _ _ h => Quot.sound (powersetAux_perm h))


theorem powerset_coe (l : List α) : @powerset α l = ((sublists l).map (↑) : List (Multiset α)) :=
  congr_arg ((↑) : List (Multiset α) → Multiset (Multiset α)) powersetAux_eq_map_coe


@[simp]
theorem powerset_coe' (l : List α) : @powerset α l = ((sublists' l).map (↑) : List (Multiset α)) :=
  Quot.sound powersetAux_perm_powersetAux'


@[simp]
theorem powerset_zero : @powerset α 0 = {0} :=
  rfl


@[simp]
theorem powerset_cons (a : α) (s) : powerset (a ::ₘ s) = powerset s + map (cons a) (powerset s) :=
                                     /-
                                       α : Type u_1
                                       a : α
                                       s : Multiset α
                                       l : List α
                                       ⊢ Eq (Multiset.cons a (Quotient.mk (List.isSetoid α) l)).powerset (HAdd.hAdd ( …
                                     -/
  Quotient.inductionOn s fun l => by simp [Function.comp_def]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem mem_powerset {s t : Multiset α} : s ∈ powerset t ↔ s ≤ t :=
                                  /-
                                    α : Type u_1
                                    s t : Multiset α
                                    ⊢ ∀ (a b : List α), Iff (Membership.mem (Multiset.powerset (Quotient.mk (List. …
                                  -/
  Quotient.inductionOn₂ s t <| by simp [Subperm, and_comm]
                                  /-
                                    🎉 no goals
                                  -/


theorem map_single_le_powerset (s : Multiset α) : s.map singleton ≤ powerset s :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      s : Multiset α
      l : List α
      ⊢ LE.le (Multiset.map Singleton.singleton (Quotient.mk (List.isSetoid α) l)) ( …
    -/
    simp only [powerset_coe, quot_mk_to_coe, coe_le, map_coe]
    /-
      α : Type u_1
      s : Multiset α
      l : List α
      ⊢ (List.map Singleton.singleton l).Subperm (List.map Multiset.ofList l.sublists)
    -/
    show l.map (((↑) : List α → Multiset α) ∘ pure) <+~ (sublists l).map (↑)
    /-
      α : Type u_1
      s : Multiset α
      l : List α
      ⊢ (List.map (Function.comp Multiset.ofList Pure.pure) l).Subperm (List.map Mul …
    -/
    rw [← List.map_map]
    /-
      α : Type u_1
      s : Multiset α
      l : List α
      ⊢ (List.map Multiset.ofList (List.map Pure.pure l)).Subperm (List.map Multiset …
    -/
    exact ((map_pure_sublist_sublists _).map _).subperm
    /-
      🎉 no goals
    -/


@[simp]
theorem card_powerset (s : Multiset α) : card (powerset s) = 2 ^ card s :=
                               /-
                                 α : Type u_1
                                 s : Multiset α
                                 ⊢ ∀ (a : List α), Eq (Multiset.powerset (Quotient.mk (List.isSetoid α) a)).car …
                               -/
  Quotient.inductionOn s <| by simp
                               /-
                                 🎉 no goals
                               -/


theorem revzip_powersetAux {l : List α} ⦃x⦄ (h : x ∈ revzip (powersetAux l)) : x.1 + x.2 = ↑l := by
  /-
    α : Type u_1
    l : List α
    x : Prod (Multiset α) (Multiset α)
    h : Membership.mem (Multiset.powersetAux l).revzip x
    ⊢ Eq (HAdd.hAdd x.1 x.2) ↑l
  -/
  rw [revzip, powersetAux_eq_map_coe, ← map_reverse, zip_map, ← revzip, List.mem_map] at h
  /-
    α : Type u_1
    l : List α
    x : Prod (Multiset α) (Multiset α)
    h : Exists fun a => And (Membership.mem l.sublists.revzip a) (Eq (Prod.map Mul …
    ⊢ Eq (HAdd.hAdd x.1 x.2) ↑l
  -/
  simp only [Prod.map_apply, Prod.exists] at h
  /-
    α : Type u_1
    l : List α
    x : Prod (Multiset α) (Multiset α)
    h : Exists fun a => Exists fun b => And (Membership.mem l.sublists.revzip { fs …
    ⊢ Eq (HAdd.hAdd x.1 x.2) ↑l
  -/
  rcases h with ⟨l₁, l₂, h, rfl, rfl⟩
  /-
    case intro.intro.intro.refl
    α : Type u_1
    l l₁ l₂ : List α
    h : Membership.mem l.sublists.revzip { fst := l₁, snd := l₂ }
    ⊢ Eq (HAdd.hAdd { fst := ↑l₁, snd := ↑l₂ }.1 { fst := ↑l₁, snd := ↑l₂ }.2) ↑l
  -/
  exact Quot.sound (revzip_sublists _ _ _ h)
  /-
    🎉 no goals
  -/


theorem revzip_powersetAux' {l : List α} ⦃x⦄ (h : x ∈ revzip (powersetAux' l)) :
    x.1 + x.2 = ↑l := by
  /-
    α : Type u_1
    l : List α
    x : Prod (Multiset α) (Multiset α)
    h : Membership.mem (Multiset.powersetAux' l).revzip x
    ⊢ Eq (HAdd.hAdd x.1 x.2) ↑l
  -/
  rw [revzip, powersetAux', ← map_reverse, zip_map, ← revzip, List.mem_map] at h
  /-
    α : Type u_1
    l : List α
    x : Prod (Multiset α) (Multiset α)
    h : Exists fun a => And (Membership.mem l.sublists'.revzip a) (Eq (Prod.map Mu …
    ⊢ Eq (HAdd.hAdd x.1 x.2) ↑l
  -/
  simp only [Prod.map_apply, Prod.exists] at h
  /-
    α : Type u_1
    l : List α
    x : Prod (Multiset α) (Multiset α)
    h : Exists fun a => Exists fun b => And (Membership.mem l.sublists'.revzip { f …
    ⊢ Eq (HAdd.hAdd x.1 x.2) ↑l
  -/
  rcases h with ⟨l₁, l₂, h, rfl, rfl⟩
  /-
    case intro.intro.intro.refl
    α : Type u_1
    l l₁ l₂ : List α
    h : Membership.mem l.sublists'.revzip { fst := l₁, snd := l₂ }
    ⊢ Eq (HAdd.hAdd { fst := ↑l₁, snd := ↑l₂ }.1 { fst := ↑l₁, snd := ↑l₂ }.2) ↑l
  -/
  exact Quot.sound (revzip_sublists' _ _ _ h)
  /-
    🎉 no goals
  -/


theorem revzip_powersetAux_lemma {α : Type*} [DecidableEq α] (l : List α) {l' : List (Multiset α)}
    (H : ∀ ⦃x : _ × _⦄, x ∈ revzip l' → x.1 + x.2 = ↑l) :
    revzip l' = l'.map fun x => (x, (l : Multiset α) - x) := by
  have :
    Forall₂ (fun (p : Multiset α × Multiset α) (s : Multiset α) => p = (s, ↑l - s)) (revzip l')
      ((revzip l').map Prod.fst) := by
    rw [forall₂_map_right_iff, forall₂_same]
    rintro ⟨s, t⟩ h
    dsimp
    rw [← H h, add_tsub_cancel_left]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    l' : List (Multiset α)
    H : ∀ ⦃x : Prod (Multiset α) (Multiset α)⦄, Membership.mem l'.revzip x → Eq (H …
    this : List.Forall₂ (fun p s => Eq p { fst := s, snd := HSub.hSub (↑l) s }) l' …
    ⊢ Eq l'.revzip (List.map (fun x => { fst := x, snd := HSub.hSub (↑l) x }) l')
  -/
  rw [← forall₂_eq_eq_eq, forall₂_map_right_iff]
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    l : List α
    l' : List (Multiset α)
    H : ∀ ⦃x : Prod (Multiset α) (Multiset α)⦄, Membership.mem l'.revzip x → Eq (H …
    this : List.Forall₂ (fun p s => Eq p { fst := s, snd := HSub.hSub (↑l) s }) l' …
    ⊢ List.Forall₂ (fun a c => Eq a { fst := c, snd := HSub.hSub (↑l) c }) l'.revz …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


theorem revzip_powersetAux_perm_aux' {l : List α} :
    revzip (powersetAux l) ~ revzip (powersetAux' l) := by
  /-
    α : Type u_1
    l : List α
    ⊢ (Multiset.powersetAux l).revzip.Perm (Multiset.powersetAux' l).revzip
  -/
  haveI := Classical.decEq α
  /-
    α : Type u_1
    l : List α
    this : DecidableEq α
    ⊢ (Multiset.powersetAux l).revzip.Perm (Multiset.powersetAux' l).revzip
  -/
  rw [revzip_powersetAux_lemma l revzip_powersetAux, revzip_powersetAux_lemma l revzip_powersetAux']
  /-
    α : Type u_1
    l : List α
    this : DecidableEq α
    ⊢ (List.map (fun x => { fst := x, snd := HSub.hSub (↑l) x }) (Multiset.powerse …
  -/
  exact powersetAux_perm_powersetAux'.map _
  /-
    🎉 no goals
  -/


theorem revzip_powersetAux_perm {l₁ l₂ : List α} (p : l₁ ~ l₂) :
    revzip (powersetAux l₁) ~ revzip (powersetAux l₂) := by
  /-
    α : Type u_1
    l₁ l₂ : List α
    p : l₁.Perm l₂
    ⊢ (Multiset.powersetAux l₁).revzip.Perm (Multiset.powersetAux l₂).revzip
  -/
  haveI := Classical.decEq α
  /-
    α : Type u_1
    l₁ l₂ : List α
    p : l₁.Perm l₂
    this : DecidableEq α
    ⊢ (Multiset.powersetAux l₁).revzip.Perm (Multiset.powersetAux l₂).revzip
  -/
  simp only [fun l : List α => revzip_powersetAux_lemma l revzip_powersetAux, coe_eq_coe.2 p]
  /-
    α : Type u_1
    l₁ l₂ : List α
    p : l₁.Perm l₂
    this : DecidableEq α
    ⊢ (List.map (fun x => { fst := x, snd := HSub.hSub (↑l₂) x }) (Multiset.powers …
  -/
  exact (powersetAux_perm p).map _
  /-
    🎉 no goals
  -/


/-- Helper function for `powersetCard`. Given a list `l`, `powersetCardAux n l` is the list
of sublists of length `n`, as multisets. -/
def powersetCardAux (n : ℕ) (l : List α) : List (Multiset α) :=
  sublistsLenAux n l (↑) []


theorem powersetCardAux_eq_map_coe {n} {l : List α} :
    powersetCardAux n l = (sublistsLen n l).map (↑) := by
  /-
    α : Type u_1
    n : Nat
    l : List α
    ⊢ Eq (Multiset.powersetCardAux n l) (List.map Multiset.ofList (List.sublistsLe …
  -/
  rw [powersetCardAux, sublistsLenAux_eq, append_nil]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_powersetCardAux {n} {l : List α} {s} : s ∈ powersetCardAux n l ↔ s ≤ ↑l ∧ card s = n :=
  Quotient.inductionOn s <| by
    simp only [quot_mk_to_coe, powersetCardAux_eq_map_coe, List.mem_map, mem_sublistsLen,
      coe_eq_coe, coe_le, Subperm, exists_prop, coe_card]
    exact fun l₁ =>
      ⟨fun ⟨l₂, ⟨s, e⟩, p⟩ => ⟨⟨_, p, s⟩, p.symm.length_eq.trans e⟩,
       fun ⟨⟨l₂, p, s⟩, e⟩ => ⟨_, ⟨s, p.length_eq.trans e⟩, p⟩⟩


@[simp]
theorem powersetCardAux_zero (l : List α) : powersetCardAux 0 l = [0] := by
  /-
    α : Type u_1
    l : List α
    ⊢ Eq (Multiset.powersetCardAux 0 l) (List.cons 0 List.nil)
  -/
  simp [powersetCardAux_eq_map_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem powersetCardAux_nil (n : ℕ) : powersetCardAux (n + 1) (@nil α) = [] :=
  rfl


@[simp]
theorem powersetCardAux_cons (n : ℕ) (a : α) (l : List α) :
    powersetCardAux (n + 1) (a :: l) =
      powersetCardAux (n + 1) l ++ List.map (cons a) (powersetCardAux n l) := by
  /-
    α : Type u_1
    n : Nat
    a : α
    l : List α
    ⊢ Eq (Multiset.powersetCardAux (HAdd.hAdd n 1) (List.cons a l)) (HAppend.hAppe …
  -/
  simp [powersetCardAux_eq_map_coe]
  /-
    🎉 no goals
  -/


theorem powersetCardAux_perm {n} {l₁ l₂ : List α} (p : l₁ ~ l₂) :
    powersetCardAux n l₁ ~ powersetCardAux n l₂ := by
  /-
    α : Type u_1
    n : Nat
    l₁ l₂ : List α
    p : l₁.Perm l₂
    ⊢ (Multiset.powersetCardAux n l₁).Perm (Multiset.powersetCardAux n l₂)
  -/
  induction' n with n IHn generalizing l₁ l₂
    /-
      case zero
      α : Type u_1
      l₁ l₂ : List α
      p : l₁.Perm l₂
      ⊢ (Multiset.powersetCardAux 0 l₁).Perm (Multiset.powersetCardAux 0 l₂)
    -/
  · simp
    /-
      🎉 no goals
    -/
  induction p with
  | nil => rfl
  | cons _ p IH =>
    simp only [powersetCardAux_cons]
    exact IH.append ((IHn p).map _)
  | swap a b =>
    simp only [powersetCardAux_cons, append_assoc]
    apply Perm.append_left
    cases n
    · simp [Perm.swap]
    simp only [powersetCardAux_cons, map_append, List.map_map]
    rw [← append_assoc, ← append_assoc,
      (by funext s; simp [cons_swap] : cons b ∘ cons a = cons a ∘ cons b)]
    exact perm_append_comm.append_right _
  | trans _ _ IH₁ IH₂ => exact IH₁.trans IH₂


/-- `powersetCard n s` is the multiset of all submultisets of `s` of length `n`. -/
def powersetCard (n : ℕ) (s : Multiset α) : Multiset (Multiset α) :=
  Quot.liftOn s (fun l => (powersetCardAux n l : Multiset (Multiset α))) fun _ _ h =>
    Quot.sound (powersetCardAux_perm h)


theorem powersetCard_coe' (n) (l : List α) : @powersetCard α n l = powersetCardAux n l :=
  rfl


theorem powersetCard_coe (n) (l : List α) :
    @powersetCard α n l = ((sublistsLen n l).map (↑) : List (Multiset α)) :=
  congr_arg ((↑) : List (Multiset α) → Multiset (Multiset α)) powersetCardAux_eq_map_coe


@[simp]
theorem powersetCard_zero_left (s : Multiset α) : powersetCard 0 s = {0} :=
                                     /-
                                       α : Type u_1
                                       s : Multiset α
                                       l : List α
                                       ⊢ Eq (Multiset.powersetCard 0 (Quotient.mk (List.isSetoid α) l)) (Singleton.si …
                                     -/
  Quotient.inductionOn s fun l => by simp [powersetCard_coe']
                                     /-
                                       🎉 no goals
                                     -/


theorem powersetCard_zero_right (n : ℕ) : @powersetCard α (n + 1) 0 = 0 :=
  rfl


@[simp]
theorem powersetCard_cons (n : ℕ) (a : α) (s) :
    powersetCard (n + 1) (a ::ₘ s) = powersetCard (n + 1) s + map (cons a) (powersetCard n s) :=
                                     /-
                                       α : Type u_1
                                       n : Nat
                                       a : α
                                       s : Multiset α
                                       l : List α
                                       ⊢ Eq (Multiset.powersetCard (HAdd.hAdd n 1) (Multiset.cons a (Quotient.mk (Lis …
                                     -/
  Quotient.inductionOn s fun l => by simp [powersetCard_coe']
                                     /-
                                       🎉 no goals
                                     -/


theorem powersetCard_one (s : Multiset α) : powersetCard 1 s = s.map singleton :=
  Quotient.inductionOn s fun l ↦ by
    /-
      α : Type u_1
      s : Multiset α
      l : List α
      ⊢ Eq (Multiset.powersetCard 1 (Quotient.mk (List.isSetoid α) l)) (Multiset.map …
    -/
    simp [powersetCard_coe, sublistsLen_one, map_reverse, Function.comp_def]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_powersetCard {n : ℕ} {s t : Multiset α} : s ∈ powersetCard n t ↔ s ≤ t ∧ card s = n :=
                                     /-
                                       α : Type u_1
                                       n : Nat
                                       s t : Multiset α
                                       l : List α
                                       ⊢ Iff (Membership.mem (Multiset.powersetCard n (Quotient.mk (List.isSetoid α)  …
                                     -/
  Quotient.inductionOn t fun l => by simp [powersetCard_coe']
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem card_powersetCard (n : ℕ) (s : Multiset α) :
    card (powersetCard n s) = Nat.choose (card s) n :=
                               /-
                                 α : Type u_1
                                 n : Nat
                                 s : Multiset α
                                 ⊢ ∀ (a : List α), Eq (Multiset.powersetCard n (Quotient.mk (List.isSetoid α) a …
                               -/
  Quotient.inductionOn s <| by simp [powersetCard_coe]
                               /-
                                 🎉 no goals
                               -/


theorem powersetCard_le_powerset (n : ℕ) (s : Multiset α) : powersetCard n s ≤ powerset s :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      n : Nat
      s : Multiset α
      l : List α
      ⊢ LE.le (Multiset.powersetCard n (Quotient.mk (List.isSetoid α) l)) (Multiset. …
    -/
    simp only [quot_mk_to_coe, powersetCard_coe, powerset_coe', coe_le]
    /-
      α : Type u_1
      n : Nat
      s : Multiset α
      l : List α
      ⊢ (List.map Multiset.ofList (List.sublistsLen n l)).Subperm (List.map Multiset …
    -/
    exact ((sublistsLen_sublist_sublists' _ _).map _).subperm
    /-
      🎉 no goals
    -/


theorem powersetCard_mono (n : ℕ) {s t : Multiset α} (h : s ≤ t) :
    powersetCard n s ≤ powersetCard n t :=
  leInductionOn h fun {l₁ l₂} h => by
    /-
      α : Type u_1
      n : Nat
      s t : Multiset α
      h✝ : LE.le s t
      l₁ l₂ : List α
      h : l₁.Sublist l₂
      ⊢ LE.le (Multiset.powersetCard n ↑l₁) (Multiset.powersetCard n ↑l₂)
    -/
    simp only [powersetCard_coe, coe_le]
    /-
      α : Type u_1
      n : Nat
      s t : Multiset α
      h✝ : LE.le s t
      l₁ l₂ : List α
      h : l₁.Sublist l₂
      ⊢ (List.map Multiset.ofList (List.sublistsLen n l₁)).Subperm (List.map Multise …
    -/
    exact ((sublistsLen_sublist_of_sublist _ h).map _).subperm
    /-
      🎉 no goals
    -/


@[simp]
theorem powersetCard_eq_empty {α : Type*} (n : ℕ) {s : Multiset α} (h : card s < n) :
    powersetCard n s = 0 :=
  card_eq_zero.mp (Nat.choose_eq_zero_of_lt h ▸ card_powersetCard _ _)


@[simp]
theorem powersetCard_card_add (s : Multiset α) {i : ℕ} (hi : 0 < i) :
    s.powersetCard (card s + i) = 0 :=
  powersetCard_eq_empty _ (Nat.lt_add_of_pos_right hi)


theorem powersetCard_map {β : Type*} (f : α → β) (n : ℕ) (s : Multiset α) :
    powersetCard n (s.map f) = (powersetCard n s).map (map f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    n : Nat
    s : Multiset α
    ⊢ Eq (Multiset.powersetCard n (Multiset.map f s)) (Multiset.map (Multiset.map  …
  -/
  induction' s using Multiset.induction with t s ih generalizing n
    /-
      case empty
      α : Type u_1
      β : Type u_2
      f : α → β
      n : Nat
      ⊢ Eq (Multiset.powersetCard n (Multiset.map f 0)) (Multiset.map (Multiset.map  …
    -/
                /-
                  🎉 no goals
                -/
  · cases n <;> simp [powersetCard_zero_left, powersetCard_zero_right]
                /-
                  🎉 no goals
                -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      f : α → β
      t : α
      s : Multiset α
      ih : ∀ (n : Nat), Eq (Multiset.powersetCard n (Multiset.map f s)) (Multiset.ma …
      n : Nat
      ⊢ Eq (Multiset.powersetCard n (Multiset.map f (Multiset.cons t s))) (Multiset. …
    -/
                /-
                  🎉 no goals
                -/
  · cases n <;> simp [ih, map_comp_cons]
                /-
                  🎉 no goals
                -/


theorem pairwise_disjoint_powersetCard (s : Multiset α) :
    _root_.Pairwise fun i j => Disjoint (s.powersetCard i) (s.powersetCard j) :=
  fun _ _ h ↦ disjoint_left.mpr fun hi hj ↦
    h ((Multiset.mem_powersetCard.mp hi).2.symm.trans (Multiset.mem_powersetCard.mp hj).2)


theorem bind_powerset_len {α : Type*} (S : Multiset α) :
    (bind (Multiset.range (card S + 1)) fun k => S.powersetCard k) = S.powerset := by
  /-
    α : Type u_2
    S : Multiset α
    ⊢ Eq ((Multiset.range (HAdd.hAdd S.card 1)).bind fun k => Multiset.powersetCar …
  -/
  induction S using Quotient.inductionOn
  simp_rw [quot_mk_to_coe, powerset_coe', powersetCard_coe, ← coe_range, coe_bind,
    ← List.map_flatMap, coe_card]
  /-
    case h
    α : Type u_2
    a✝ : List α
    ⊢ Eq ↑(List.map Multiset.ofList ((List.range (HAdd.hAdd a✝.length 1)).flatMap  …
  -/
  exact coe_eq_coe.mpr ((List.range_bind_sublistsLen_perm _).map _)
  /-
    🎉 no goals
  -/


@[simp]
theorem nodup_powerset {s : Multiset α} : Nodup (powerset s) ↔ Nodup s :=
  ⟨fun h => (nodup_of_le (map_single_le_powerset _) h).of_map _,
    Quotient.inductionOn s fun l h => by
      /-
        α : Type u_1
        s : Multiset α
        l : List α
        h : Multiset.Nodup (Quotient.mk (List.isSetoid α) l)
        ⊢ (Multiset.powerset (Quotient.mk (List.isSetoid α) l)).Nodup
      -/
      simp only [quot_mk_to_coe, powerset_coe', coe_nodup]
      /-
        α : Type u_1
        s : Multiset α
        l : List α
        h : Multiset.Nodup (Quotient.mk (List.isSetoid α) l)
        ⊢ (List.map Multiset.ofList l.sublists').Nodup
      -/
      refine (nodup_sublists'.2 h).map_on ?_
      exact fun x sx y sy e =>
        (h.perm_iff_eq_of_sublist (mem_sublists'.1 sx) (mem_sublists'.1 sy)).1 (Quotient.exact e)⟩


alias ⟨Nodup.ofPowerset, Nodup.powerset⟩ := nodup_powerset


protected theorem Nodup.powersetCard {n : ℕ} {s : Multiset α} (h : Nodup s) :
    Nodup (powersetCard n s) :=
  nodup_of_le (powersetCard_le_powerset _ _) (nodup_powerset.2 h)


