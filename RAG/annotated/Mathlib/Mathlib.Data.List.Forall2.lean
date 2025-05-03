mk_iff_of_inductive_prop List.Forall₂ List.forall₂_iff


theorem Forall₂.imp (H : ∀ a b, R a b → S a b) {l₁ l₂} (h : Forall₂ R l₁ l₂) : Forall₂ S l₁ l₂ := by
  /-
    α : Type u_1
    β : Type u_2
    R S : α → β → Prop
    H : ∀ (a : α) (b : β), R a b → S a b
    l₁ : List α
    l₂ : List β
    h : List.Forall₂ R l₁ l₂
    ⊢ List.Forall₂ S l₁ l₂
  -/
                  /-
                    🎉 no goals
                  -/
                                  /-
                                    🎉 no goals
                                  -/
  induction h <;> constructor <;> solve_by_elim
                                  /-
                                    🎉 no goals
                                  -/


theorem Forall₂.mp {Q : α → β → Prop} (h : ∀ a b, Q a b → R a b → S a b) :
    ∀ {l₁ l₂}, Forall₂ Q l₁ l₂ → Forall₂ R l₁ l₂ → Forall₂ S l₁ l₂
  | [], [], Forall₂.nil, Forall₂.nil => Forall₂.nil
  | a :: _, b :: _, Forall₂.cons hr hrs, Forall₂.cons hq hqs =>
    Forall₂.cons (h a b hr hq) (Forall₂.mp h hrs hqs)


theorem Forall₂.flip : ∀ {a b}, Forall₂ (flip R) b a → Forall₂ R a b
  | _, _, Forall₂.nil => Forall₂.nil
  | _ :: _, _ :: _, Forall₂.cons h₁ h₂ => Forall₂.cons h₁ h₂.flip


@[simp]
theorem forall₂_same : ∀ {l : List α}, Forall₂ Rₐ l l ↔ ∀ x ∈ l, Rₐ x x
             /-
               α : Type u_1
               Rₐ : α → α → Prop
               ⊢ Iff (List.Forall₂ Rₐ List.nil List.nil) (∀ (x : α), Membership.mem List.nil  …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                 /-
                   α : Type u_1
                   Rₐ : α → α → Prop
                   a : α
                   l : List α
                   ⊢ Iff (List.Forall₂ Rₐ (List.cons a l) (List.cons a l)) (∀ (x : α), Membership …
                 -/
  | a :: l => by simp [@forall₂_same l]
                 /-
                   🎉 no goals
                 -/


theorem forall₂_refl [IsRefl α Rₐ] (l : List α) : Forall₂ Rₐ l l :=
  forall₂_same.2 fun _ _ => refl _


@[simp]
theorem forall₂_eq_eq_eq : Forall₂ ((· = ·) : α → α → Prop) = Eq := by
  /-
    α : Type u_1
    ⊢ Eq (List.Forall₂ fun x1 x2 => Eq x1 x2) Eq
  -/
  funext a b; apply propext
  /-
    case h.h.a
    α : Type u_1
    a b : List α
    ⊢ Iff (List.Forall₂ (fun x1 x2 => Eq x1 x2) a b) (Eq a b)
  -/
  constructor
    /-
      case h.h.a.mp
      α : Type u_1
      a b : List α
      ⊢ List.Forall₂ (fun x1 x2 => Eq x1 x2) a b → Eq a b
    -/
  · intro h
    /-
      case h.h.a.mp
      α : Type u_1
      a b : List α
      h : List.Forall₂ (fun x1 x2 => Eq x1 x2) a b
      ⊢ Eq a b
    -/
    induction h
      /-
        case h.h.a.mp.nil
        α : Type u_1
        a b : List α
        ⊢ Eq List.nil List.nil
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case h.h.a.mp.cons
      α : Type u_1
      a b : List α
      a✝² b✝ : α
      l₁✝ l₂✝ : List α
      a✝¹ : Eq a✝² b✝
      a✝ : List.Forall₂ (fun x1 x2 => Eq x1 x2) l₁✝ l₂✝
      a_ih✝ : Eq l₁✝ l₂✝
      ⊢ Eq (List.cons a✝² l₁✝) (List.cons b✝ l₂✝)
    -/
    simp only [*]
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.mpr
      α : Type u_1
      a b : List α
      ⊢ Eq a b → List.Forall₂ (fun x1 x2 => Eq x1 x2) a b
    -/
  · rintro rfl
    /-
      case h.h.a.mpr
      α : Type u_1
      a : List α
      ⊢ List.Forall₂ (fun x1 x2 => Eq x1 x2) a a
    -/
    exact forall₂_refl _
    /-
      🎉 no goals
    -/


@[simp]
theorem forall₂_nil_left_iff {l} : Forall₂ R nil l ↔ l = nil :=
               /-
                 α : Type u_1
                 β : Type u_2
                 R : α → β → Prop
                 l : List β
                 H : List.Forall₂ R List.nil l
                 ⊢ Eq l List.nil
               -/
                        /-
                          🎉 no goals
                        -/
  ⟨fun H => by cases H; rfl, by rintro rfl; exact Forall₂.nil⟩
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem forall₂_nil_right_iff {l} : Forall₂ R l nil ↔ l = nil :=
               /-
                 α : Type u_1
                 β : Type u_2
                 R : α → β → Prop
                 l : List α
                 H : List.Forall₂ R l List.nil
                 ⊢ Eq l List.nil
               -/
                        /-
                          🎉 no goals
                        -/
  ⟨fun H => by cases H; rfl, by rintro rfl; exact Forall₂.nil⟩
                                            /-
                                              🎉 no goals
                                            -/


theorem forall₂_cons_left_iff {a l u} :
    Forall₂ R (a :: l) u ↔ ∃ b u', R a b ∧ Forall₂ R l u' ∧ u = b :: u' :=
  Iff.intro
    (fun h =>
      match u, h with
      | b :: u', Forall₂.cons h₁ h₂ => ⟨b, u', h₁, h₂, rfl⟩)
    fun h =>
    match u, h with
    | _, ⟨_, _, h₁, h₂, rfl⟩ => Forall₂.cons h₁ h₂


theorem forall₂_cons_right_iff {b l u} :
    Forall₂ R u (b :: l) ↔ ∃ a u', R a b ∧ Forall₂ R u' l ∧ u = a :: u' :=
  Iff.intro
    (fun h =>
      match u, h with
      | b :: u', Forall₂.cons h₁ h₂ => ⟨b, u', h₁, h₂, rfl⟩)
    fun h =>
    match u, h with
    | _, ⟨_, _, h₁, h₂, rfl⟩ => Forall₂.cons h₁ h₂


theorem forall₂_and_left {p : α → Prop} :
    ∀ l u, Forall₂ (fun a b => p a ∧ R a b) l u ↔ (∀ a ∈ l, p a) ∧ Forall₂ R l u
  | [], u => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      p : α → Prop
      u : List β
      ⊢ Iff (List.Forall₂ (fun a b => And (p a) (R a b)) List.nil u) (And (∀ (a : α) …
    -/
    simp only [forall₂_nil_left_iff, forall_prop_of_false (not_mem_nil _), imp_true_iff, true_and]
    /-
      🎉 no goals
    -/
  | a :: l, u => by
    simp only [forall₂_and_left l, forall₂_cons_left_iff, forall_mem_cons, and_assoc,
      @and_comm _ (p a), @and_left_comm _ (p a), exists_and_left]
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      p : α → Prop
      a : α
      l : List α
      u : List β
      ⊢ Iff (And (p a) (Exists fun x => And (R a x) (And (∀ (a : α), Membership.mem  …
    -/
    simp only [and_comm, and_assoc, and_left_comm, ← exists_and_right]
    /-
      🎉 no goals
    -/


@[simp]
theorem forall₂_map_left_iff {f : γ → α} :
    ∀ {l u}, Forall₂ R (map f l) u ↔ Forall₂ (fun c b => R (f c) b) l u
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  R : α → β → Prop
                  f : γ → α
                  x✝ : List β
                  ⊢ Iff (List.Forall₂ R (List.map f List.nil) x✝) (List.Forall₂ (fun c b => R (f …
                -/
  | [], _ => by simp only [map, forall₂_nil_left_iff]
                /-
                  🎉 no goals
                -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      R : α → β → Prop
                      f : γ → α
                      a : γ
                      l : List γ
                      x✝ : List β
                      ⊢ Iff (List.Forall₂ R (List.map f (List.cons a l)) x✝) (List.Forall₂ (fun c b  …
                    -/
  | a :: l, _ => by simp only [map, forall₂_cons_left_iff, forall₂_map_left_iff]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem forall₂_map_right_iff {f : γ → β} :
    ∀ {l u}, Forall₂ R l (map f u) ↔ Forall₂ (fun a c => R a (f c)) l u
                /-
                  α : Type u_1
                  β : Type u_2
                  γ : Type u_3
                  R : α → β → Prop
                  f : γ → β
                  x✝ : List α
                  ⊢ Iff (List.Forall₂ R x✝ (List.map f List.nil)) (List.Forall₂ (fun a c => R a  …
                -/
  | _, [] => by simp only [map, forall₂_nil_right_iff]
                /-
                  🎉 no goals
                -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      R : α → β → Prop
                      f : γ → β
                      x✝ : List α
                      b : γ
                      u : List γ
                      ⊢ Iff (List.Forall₂ R x✝ (List.map f (List.cons b u))) (List.Forall₂ (fun a c  …
                    -/
  | _, b :: u => by simp only [map, forall₂_cons_right_iff, forall₂_map_right_iff]
                    /-
                      🎉 no goals
                    -/


theorem left_unique_forall₂' (hr : LeftUnique R) : ∀ {a b c}, Forall₂ R a c → Forall₂ R b c → a = b
  | _, _, _, Forall₂.nil, Forall₂.nil => rfl
  | _, _, _, Forall₂.cons ha₀ h₀, Forall₂.cons ha₁ h₁ =>
    hr ha₀ ha₁ ▸ left_unique_forall₂' hr h₀ h₁ ▸ rfl


theorem _root_.Relator.LeftUnique.forall₂ (hr : LeftUnique R) : LeftUnique (Forall₂ R) :=
  @left_unique_forall₂' _ _ _ hr


theorem right_unique_forall₂' (hr : RightUnique R) :
    ∀ {a b c}, Forall₂ R a b → Forall₂ R a c → b = c
  | _, _, _, Forall₂.nil, Forall₂.nil => rfl
  | _, _, _, Forall₂.cons ha₀ h₀, Forall₂.cons ha₁ h₁ =>
    hr ha₀ ha₁ ▸ right_unique_forall₂' hr h₀ h₁ ▸ rfl


theorem _root_.Relator.RightUnique.forall₂ (hr : RightUnique R) : RightUnique (Forall₂ R) :=
  @right_unique_forall₂' _ _ _ hr


theorem _root_.Relator.BiUnique.forall₂ (hr : BiUnique R) : BiUnique (Forall₂ R) :=
  ⟨hr.left.forall₂, hr.right.forall₂⟩


theorem Forall₂.length_eq : ∀ {l₁ l₂}, Forall₂ R l₁ l₂ → length l₁ = length l₂
  | _, _, Forall₂.nil => rfl
  | _, _, Forall₂.cons _ h₂ => congr_arg succ (Forall₂.length_eq h₂)


theorem Forall₂.get :
    ∀ {x : List α} {y : List β}, Forall₂ R x y →
      ∀ ⦃i : ℕ⦄ (hx : i < x.length) (hy : i < y.length), R (x.get ⟨i, hx⟩) (y.get ⟨i, hy⟩)
  | _, _, Forall₂.cons ha _, 0, _, _ => ha
  | _, _, Forall₂.cons _ hl, succ _, _, _ => hl.get _ _


theorem forall₂_of_length_eq_of_get :
    ∀ {x : List α} {y : List β},
      x.length = y.length → (∀ i h₁ h₂, R (x.get ⟨i, h₁⟩) (y.get ⟨i, h₂⟩)) → Forall₂ R x y
  | [], [], _, _ => Forall₂.nil
  | _ :: _, _ :: _, hl, h =>
    Forall₂.cons (h 0 (Nat.zero_lt_succ _) (Nat.zero_lt_succ _))
      (forall₂_of_length_eq_of_get (succ.inj hl) fun i h₁ h₂ =>
        h i.succ (succ_lt_succ h₁) (succ_lt_succ h₂))


theorem forall₂_iff_get {l₁ : List α} {l₂ : List β} :
    Forall₂ R l₁ l₂ ↔ l₁.length = l₂.length ∧ ∀ i h₁ h₂, R (l₁.get ⟨i, h₁⟩) (l₂.get ⟨i, h₂⟩) :=
  ⟨fun h => ⟨h.length_eq, h.get⟩, fun h => forall₂_of_length_eq_of_get h.1 h.2⟩


theorem forall₂_zip : ∀ {l₁ l₂}, Forall₂ R l₁ l₂ → ∀ {a b}, (a, b) ∈ zip l₁ l₂ → R a b
  | _, _, Forall₂.cons h₁ h₂, x, y, hx => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      a✝ : α
      b✝ : β
      l₁✝ : List α
      l₂✝ : List β
      h₁ : R a✝ b✝
      h₂ : List.Forall₂ R l₁✝ l₂✝
      x : α
      y : β
      hx : Membership.mem ((List.cons a✝ l₁✝).zip (List.cons b✝ l₂✝)) { fst := x, sn …
      ⊢ R x y
    -/
    rw [zip, zipWith, mem_cons] at hx
    match hx with
    | Or.inl rfl => exact h₁
    | Or.inr h₃ => exact forall₂_zip h₂ h₃


theorem forall₂_iff_zip {l₁ l₂} :
    Forall₂ R l₁ l₂ ↔ length l₁ = length l₂ ∧ ∀ {a b}, (a, b) ∈ zip l₁ l₂ → R a b :=
  ⟨fun h => ⟨Forall₂.length_eq h, @forall₂_zip _ _ _ _ _ h⟩, fun h => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      l₁ : List α
      l₂ : List β
      h : And (Eq l₁.length l₂.length) (∀ {a : α} {b : β}, Membership.mem (l₁.zip l₂ …
      ⊢ List.Forall₂ R l₁ l₂
    -/
    cases' h with h₁ h₂
    /-
      case intro
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      l₁ : List α
      l₂ : List β
      h₁ : Eq l₁.length l₂.length
      h₂ : ∀ {a : α} {b : β}, Membership.mem (l₁.zip l₂) { fst := a, snd := b } → R  …
      ⊢ List.Forall₂ R l₁ l₂
    -/
    induction' l₁ with a l₁ IH generalizing l₂
      /-
        case intro.nil
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        l₂ : List β
        h₁ : Eq List.nil.length l₂.length
        h₂ : ∀ {a : α} {b : β}, Membership.mem (List.nil.zip l₂) { fst := a, snd := b  …
        ⊢ List.Forall₂ R List.nil l₂
      -/
    · cases length_eq_zero.1 h₁.symm
      /-
        case intro.nil.refl
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        h₁ : Eq List.nil.length List.nil.length
        h₂ : ∀ {a : α} {b : β}, Membership.mem (List.nil.zip List.nil) { fst := a, snd …
        ⊢ List.Forall₂ R List.nil List.nil
      -/
      constructor
      /-
        🎉 no goals
      -/
      /-
        case intro.cons
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        a : α
        l₁ : List α
        IH : ∀ {l₂ : List β}, Eq l₁.length l₂.length → (∀ {a : α} {b : β}, Membership. …
        l₂ : List β
        h₁ : Eq (List.cons a l₁).length l₂.length
        h₂ : ∀ {a_1 : α} {b : β}, Membership.mem ((List.cons a l₁).zip l₂) { fst := a_ …
        ⊢ List.Forall₂ R (List.cons a l₁) l₂
      -/
    · cases' l₂ with b l₂
        /-
          case intro.cons.nil
          α : Type u_1
          β : Type u_2
          R : α → β → Prop
          a : α
          l₁ : List α
          IH : ∀ {l₂ : List β}, Eq l₁.length l₂.length → (∀ {a : α} {b : β}, Membership. …
          h₁ : Eq (List.cons a l₁).length List.nil.length
          h₂ : ∀ {a_1 : α} {b : β}, Membership.mem ((List.cons a l₁).zip List.nil) { fst …
          ⊢ List.Forall₂ R (List.cons a l₁) List.nil
        -/
      · simp at h₁
        /-
          🎉 no goals
        -/
        /-
          case intro.cons.cons
          α : Type u_1
          β : Type u_2
          R : α → β → Prop
          a : α
          l₁ : List α
          IH : ∀ {l₂ : List β}, Eq l₁.length l₂.length → (∀ {a : α} {b : β}, Membership. …
          b : β
          l₂ : List β
          h₁ : Eq (List.cons a l₁).length (List.cons b l₂).length
          h₂ : ∀ {a_1 : α} {b_1 : β}, Membership.mem ((List.cons a l₁).zip (List.cons b  …
          ⊢ List.Forall₂ R (List.cons a l₁) (List.cons b l₂)
        -/
      · simp only [length_cons, succ.injEq] at h₁
        exact Forall₂.cons (h₂ <| by simp [zip])
          (IH h₁ fun h => h₂ <| by
            simp only [zip, zipWith, find?, mem_cons, Prod.mk.injEq]; right
            simpa [zip] using h)⟩


theorem forall₂_take : ∀ (n) {l₁ l₂}, Forall₂ R l₁ l₂ → Forall₂ R (take n l₁) (take n l₂)
                     /-
                       α : Type u_1
                       β : Type u_2
                       R : α → β → Prop
                       x✝² : List α
                       x✝¹ : List β
                       x✝ : List.Forall₂ R x✝² x✝¹
                       ⊢ List.Forall₂ R (List.take 0 x✝²) (List.take 0 x✝¹)
                     -/
  | 0, _, _, _ => by simp only [Forall₂.nil, take]
                     /-
                       🎉 no goals
                     -/
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     R : α → β → Prop
                                     n✝ : Nat
                                     ⊢ List.Forall₂ R (List.take (HAdd.hAdd n✝ 1) List.nil) (List.take (HAdd.hAdd n …
                                   -/
  | _ + 1, _, _, Forall₂.nil => by simp only [Forall₂.nil, take]
                                   /-
                                     🎉 no goals
                                   -/
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            R : α → β → Prop
                                            n : Nat
                                            a✝ : α
                                            b✝ : β
                                            l₁✝ : List α
                                            l₂✝ : List β
                                            h₁ : R a✝ b✝
                                            h₂ : List.Forall₂ R l₁✝ l₂✝
                                            ⊢ List.Forall₂ R (List.take (HAdd.hAdd n 1) (List.cons a✝ l₁✝)) (List.take (HA …
                                          -/
  | n + 1, _, _, Forall₂.cons h₁ h₂ => by simp [And.intro h₁ h₂, forall₂_take n]
                                          /-
                                            🎉 no goals
                                          -/


theorem forall₂_drop : ∀ (n) {l₁ l₂}, Forall₂ R l₁ l₂ → Forall₂ R (drop n l₁) (drop n l₂)
                     /-
                       α : Type u_1
                       β : Type u_2
                       R : α → β → Prop
                       x✝¹ : List α
                       x✝ : List β
                       h : List.Forall₂ R x✝¹ x✝
                       ⊢ List.Forall₂ R (List.drop 0 x✝¹) (List.drop 0 x✝)
                     -/
  | 0, _, _, h => by simp only [drop, h]
                     /-
                       🎉 no goals
                     -/
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     R : α → β → Prop
                                     n✝ : Nat
                                     ⊢ List.Forall₂ R (List.drop (HAdd.hAdd n✝ 1) List.nil) (List.drop (HAdd.hAdd n …
                                   -/
  | _ + 1, _, _, Forall₂.nil => by simp only [Forall₂.nil, drop]
                                   /-
                                     🎉 no goals
                                   -/
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            R : α → β → Prop
                                            n : Nat
                                            a✝ : α
                                            b✝ : β
                                            l₁✝ : List α
                                            l₂✝ : List β
                                            h₁ : R a✝ b✝
                                            h₂ : List.Forall₂ R l₁✝ l₂✝
                                            ⊢ List.Forall₂ R (List.drop (HAdd.hAdd n 1) (List.cons a✝ l₁✝)) (List.drop (HA …
                                          -/
  | n + 1, _, _, Forall₂.cons h₁ h₂ => by simp [And.intro h₁ h₂, forall₂_drop n]
                                          /-
                                            🎉 no goals
                                          -/


theorem forall₂_take_append (l : List α) (l₁ : List β) (l₂ : List β) (h : Forall₂ R l (l₁ ++ l₂)) :
    Forall₂ R (List.take (length l₁) l) l₁ := by
  have h' : Forall₂ R (take (length l₁) l) (take (length l₁) (l₁ ++ l₂)) :=
    forall₂_take (length l₁) h
  /-
    α : Type u_1
    β : Type u_2
    R : α → β → Prop
    l : List α
    l₁ l₂ : List β
    h : List.Forall₂ R l (HAppend.hAppend l₁ l₂)
    h' : List.Forall₂ R (List.take l₁.length l) (List.take l₁.length (HAppend.hApp …
    ⊢ List.Forall₂ R (List.take l₁.length l) l₁
  -/
  rwa [take_left] at h'
  /-
    🎉 no goals
  -/


theorem forall₂_drop_append (l : List α) (l₁ : List β) (l₂ : List β) (h : Forall₂ R l (l₁ ++ l₂)) :
    Forall₂ R (List.drop (length l₁) l) l₂ := by
  have h' : Forall₂ R (drop (length l₁) l) (drop (length l₁) (l₁ ++ l₂)) :=
    forall₂_drop (length l₁) h
  /-
    α : Type u_1
    β : Type u_2
    R : α → β → Prop
    l : List α
    l₁ l₂ : List β
    h : List.Forall₂ R l (HAppend.hAppend l₁ l₂)
    h' : List.Forall₂ R (List.drop l₁.length l) (List.drop l₁.length (HAppend.hApp …
    ⊢ List.Forall₂ R (List.drop l₁.length l) l₂
  -/
  rwa [drop_left] at h'
  /-
    🎉 no goals
  -/


theorem rel_mem (hr : BiUnique R) : (R ⇒ Forall₂ R ⇒ Iff) (· ∈ ·) (· ∈ ·)
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         R : α → β → Prop
                                         hr : Relator.BiUnique R
                                         a : α
                                         b : β
                                         x✝ : R a b
                                         ⊢ Iff ((fun x1 x2 => Membership.mem x2 x1) a List.nil) ((fun x1 x2 => Membersh …
                                       -/
  | a, b, _, [], [], Forall₂.nil => by simp only [not_mem_nil]
                                       /-
                                         🎉 no goals
                                       -/
  | a, b, h, a' :: as, b' :: bs, Forall₂.cons h₁ h₂ => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      hr : Relator.BiUnique R
      a : α
      b : β
      h : R a b
      a' : α
      as : List α
      b' : β
      bs : List β
      h₁ : R a' b'
      h₂ : List.Forall₂ R as bs
      ⊢ Iff ((fun x1 x2 => Membership.mem x2 x1) a (List.cons a' as)) ((fun x1 x2 => …
    -/
    simp only [mem_cons]
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      hr : Relator.BiUnique R
      a : α
      b : β
      h : R a b
      a' : α
      as : List α
      b' : β
      bs : List β
      h₁ : R a' b'
      h₂ : List.Forall₂ R as bs
      ⊢ Iff (Or (Eq a a') (Membership.mem as a)) (Or (Eq b b') (Membership.mem bs b))
    -/
    exact rel_or (rel_eq hr h h₁) (rel_mem hr h h₂)
    /-
      🎉 no goals
    -/


theorem rel_map : ((R ⇒ P) ⇒ Forall₂ R ⇒ Forall₂ P) map map
  | _, _, _, [], [], Forall₂.nil => Forall₂.nil
  | _, _, h, _ :: _, _ :: _, Forall₂.cons h₁ h₂ => Forall₂.cons (h h₁) (rel_map (@h) h₂)


theorem rel_append : (Forall₂ R ⇒ Forall₂ R ⇒ Forall₂ R) (· ++ ·) (· ++ ·)
  | [], [], _, _, _, hl => hl
  | _, _, Forall₂.cons h₁ h₂, _, _, hl => Forall₂.cons h₁ (rel_append h₂ hl)


theorem rel_reverse : (Forall₂ R ⇒ Forall₂ R) reverse reverse
  | [], [], Forall₂.nil => Forall₂.nil
  | _, _, Forall₂.cons h₁ h₂ => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      a✝ : α
      b✝ : β
      l₁✝ : List α
      l₂✝ : List β
      h₁ : R a✝ b✝
      h₂ : List.Forall₂ R l₁✝ l₂✝
      ⊢ List.Forall₂ R (List.cons a✝ l₁✝).reverse (List.cons b✝ l₂✝).reverse
    -/
    simp only [reverse_cons]
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      a✝ : α
      b✝ : β
      l₁✝ : List α
      l₂✝ : List β
      h₁ : R a✝ b✝
      h₂ : List.Forall₂ R l₁✝ l₂✝
      ⊢ List.Forall₂ R (HAppend.hAppend l₁✝.reverse (List.cons a✝ List.nil)) (HAppen …
    -/
    exact rel_append (rel_reverse h₂) (Forall₂.cons h₁ Forall₂.nil)
    /-
      🎉 no goals
    -/


@[simp]
theorem forall₂_reverse_iff {l₁ l₂} : Forall₂ R (reverse l₁) (reverse l₂) ↔ Forall₂ R l₁ l₂ :=
  Iff.intro
    (fun h => by
      /-
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        l₁ : List α
        l₂ : List β
        h : List.Forall₂ R l₁.reverse l₂.reverse
        ⊢ List.Forall₂ R l₁ l₂
      -/
      rw [← reverse_reverse l₁, ← reverse_reverse l₂]
      /-
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        l₁ : List α
        l₂ : List β
        h : List.Forall₂ R l₁.reverse l₂.reverse
        ⊢ List.Forall₂ R l₁.reverse.reverse l₂.reverse.reverse
      -/
      exact rel_reverse h)
      /-
        🎉 no goals
      -/
    fun h => rel_reverse h


theorem rel_flatten : (Forall₂ (Forall₂ R) ⇒ Forall₂ R) flatten flatten
  | [], [], Forall₂.nil => Forall₂.nil
  | _, _, Forall₂.cons h₁ h₂ => rel_append h₁ (rel_flatten h₂)


@[deprecated (since := "2025-10-15")] alias rel_join := rel_flatten


theorem rel_flatMap : (Forall₂ R ⇒ (R ⇒ Forall₂ P) ⇒ Forall₂ P) List.flatMap List.flatMap :=
  fun _ _ h₁ _ _ h₂ => rel_flatten (rel_map (@h₂) h₁)


@[deprecated (since := "2025-10-16")] alias rel_bind := rel_flatMap


theorem rel_foldl : ((P ⇒ R ⇒ P) ⇒ P ⇒ Forall₂ R ⇒ P) foldl foldl
  | _, _, _, _, _, h, _, _, Forall₂.nil => h
  | _, _, hfg, _, _, hxy, _, _, Forall₂.cons hab hs => rel_foldl (@hfg) (hfg hxy hab) hs


theorem rel_foldr : ((R ⇒ P ⇒ P) ⇒ P ⇒ Forall₂ R ⇒ P) foldr foldr
  | _, _, _, _, _, h, _, _, Forall₂.nil => h
  | _, _, hfg, _, _, hxy, _, _, Forall₂.cons hab hs => hfg hab (rel_foldr (@hfg) hxy hs)


theorem rel_filter {p : α → Bool} {q : β → Bool}
    (hpq : (R ⇒ (· ↔ ·)) (fun x => p x) (fun x => q x)) :
    (Forall₂ R ⇒ Forall₂ R) (filter p) (filter q)
  | _, _, Forall₂.nil => Forall₂.nil
  | a :: as, b :: bs, Forall₂.cons h₁ h₂ => by
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      p : α → Bool
      q : β → Bool
      hpq : Relator.LiftFun R (fun x1 x2 => Iff x1 x2) (fun x => Eq (p x) Bool.true) …
      a : α
      as : List α
      b : β
      bs : List β
      h₁ : R a b
      h₂ : List.Forall₂ R as bs
      ⊢ List.Forall₂ R (List.filter p (List.cons a as)) (List.filter q (List.cons b  …
    -/
    dsimp [LiftFun] at hpq
    /-
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      p : α → Bool
      q : β → Bool
      hpq : ∀ ⦃a : α⦄ ⦃b : β⦄, R a b → Iff (Eq (p a) Bool.true) (Eq (q b) Bool.true)
      a : α
      as : List α
      b : β
      bs : List β
      h₁ : R a b
      h₂ : List.Forall₂ R as bs
      ⊢ List.Forall₂ R (List.filter p (List.cons a as)) (List.filter q (List.cons b  …
    -/
    by_cases h : p a
      /-
        case pos
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        p : α → Bool
        q : β → Bool
        hpq : ∀ ⦃a : α⦄ ⦃b : β⦄, R a b → Iff (Eq (p a) Bool.true) (Eq (q b) Bool.true)
        a : α
        as : List α
        b : β
        bs : List β
        h₁ : R a b
        h₂ : List.Forall₂ R as bs
        h : Eq (p a) Bool.true
        ⊢ List.Forall₂ R (List.filter p (List.cons a as)) (List.filter q (List.cons b  …
      -/
    · have : q b := by rwa [← hpq h₁]
      simp only [filter_cons_of_pos h, filter_cons_of_pos this, forall₂_cons, h₁, true_and,
        rel_filter hpq h₂]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        p : α → Bool
        q : β → Bool
        hpq : ∀ ⦃a : α⦄ ⦃b : β⦄, R a b → Iff (Eq (p a) Bool.true) (Eq (q b) Bool.true)
        a : α
        as : List α
        b : β
        bs : List β
        h₁ : R a b
        h₂ : List.Forall₂ R as bs
        h : Not (Eq (p a) Bool.true)
        ⊢ List.Forall₂ R (List.filter p (List.cons a as)) (List.filter q (List.cons b  …
      -/
    · have : ¬q b := by rwa [← hpq h₁]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        R : α → β → Prop
        p : α → Bool
        q : β → Bool
        hpq : ∀ ⦃a : α⦄ ⦃b : β⦄, R a b → Iff (Eq (p a) Bool.true) (Eq (q b) Bool.true)
        a : α
        as : List α
        b : β
        bs : List β
        h₁ : R a b
        h₂ : List.Forall₂ R as bs
        h : Not (Eq (p a) Bool.true)
        this : Not (Eq (q b) Bool.true)
        ⊢ List.Forall₂ R (List.filter p (List.cons a as)) (List.filter q (List.cons b  …
      -/
      simp only [filter_cons_of_neg h, filter_cons_of_neg this, rel_filter hpq h₂]
      /-
        🎉 no goals
      -/


theorem rel_filterMap : ((R ⇒ Option.Rel P) ⇒ Forall₂ R ⇒ Forall₂ P) filterMap filterMap
  | _, _, _, _, _, Forall₂.nil => Forall₂.nil
  | f, g, hfg, a :: as, b :: bs, Forall₂.cons h₁ h₂ => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      R : α → β → Prop
      P : γ → δ → Prop
      f : α → Option γ
      g : β → Option δ
      hfg : Relator.LiftFun R (Option.Rel P) f g
      a : α
      as : List α
      b : β
      bs : List β
      h₁ : R a b
      h₂ : List.Forall₂ R as bs
      ⊢ List.Forall₂ P (List.filterMap f (List.cons a as)) (List.filterMap g (List.c …
    -/
    rw [filterMap_cons, filterMap_cons]
    exact
      match f a, g b, hfg h₁ with
      | _, _, Option.Rel.none => rel_filterMap (@hfg) h₂
      | _, _, Option.Rel.some h => Forall₂.cons h (rel_filterMap (@hfg) h₂)


/-- Given a relation `R`, `sublist_forall₂ r l₁ l₂` indicates that there is a sublist of `l₂` such
  that `forall₂ r l₁ l₂`. -/
inductive SublistForall₂ (R : α → β → Prop) : List α → List β → Prop
  | nil {l} : SublistForall₂ R [] l
  | cons {a₁ a₂ l₁ l₂} : R a₁ a₂ → SublistForall₂ R l₁ l₂ → SublistForall₂ R (a₁ :: l₁) (a₂ :: l₂)
  | cons_right {a l₁ l₂} : SublistForall₂ R l₁ l₂ → SublistForall₂ R l₁ (a :: l₂)


theorem sublistForall₂_iff {l₁ : List α} {l₂ : List β} :
    SublistForall₂ R l₁ l₂ ↔ ∃ l, Forall₂ R l₁ l ∧ l <+ l₂ := by
  /-
    α : Type u_1
    β : Type u_2
    R : α → β → Prop
    l₁ : List α
    l₂ : List β
    ⊢ Iff (List.SublistForall₂ R l₁ l₂) (Exists fun l => And (List.Forall₂ R l₁ l) …
  -/
  constructor <;> intro h
  · induction h with
    | nil => exact ⟨nil, Forall₂.nil, nil_sublist _⟩
    | @cons a b l1 l2 rab _ ih =>
      obtain ⟨l, hl1, hl2⟩ := ih
      exact ⟨b :: l, Forall₂.cons rab hl1, hl2.cons_cons b⟩
    | cons_right _ ih =>
      obtain ⟨l, hl1, hl2⟩ := ih
      exact ⟨l, hl1, hl2.trans (Sublist.cons _ (Sublist.refl _))⟩
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      l₁ : List α
      l₂ : List β
      h : Exists fun l => And (List.Forall₂ R l₁ l) (l.Sublist l₂)
      ⊢ List.SublistForall₂ R l₁ l₂
    -/
  · obtain ⟨l, hl1, hl2⟩ := h
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      R : α → β → Prop
      l₁ : List α
      l₂ l : List β
      hl1 : List.Forall₂ R l₁ l
      hl2 : l.Sublist l₂
      ⊢ List.SublistForall₂ R l₁ l₂
    -/
    revert l₁
    induction hl2 with
    | slnil =>
      intro l₁ hl1
      rw [forall₂_nil_right_iff.1 hl1]
      exact SublistForall₂.nil
    | cons _ _ ih => intro l₁ hl1; exact SublistForall₂.cons_right (ih hl1)
    | cons₂ _ _ ih =>
      intro l₁ hl1
      cases' hl1 with _ _ _ _ hr hl _
      exact SublistForall₂.cons hr (ih hl)


instance SublistForall₂.is_refl [IsRefl α Rₐ] : IsRefl (List α) (SublistForall₂ Rₐ) :=
  ⟨fun l => sublistForall₂_iff.2 ⟨l, forall₂_refl l, Sublist.refl l⟩⟩


instance SublistForall₂.is_trans [IsTrans α Rₐ] : IsTrans (List α) (SublistForall₂ Rₐ) :=
  ⟨fun a b c => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      R S : α → β → Prop
      P : γ → δ → Prop
      Rₐ : α → α → Prop
      inst✝ : IsTrans α Rₐ
      a b c : List α
      ⊢ List.SublistForall₂ Rₐ a b → List.SublistForall₂ Rₐ b c → List.SublistForall …
    -/
    revert a b
    induction c with
    | nil =>
      rintro _ _ h1 h2
      cases h2
      exact h1
    | cons _ _ ih =>
      rintro a b h1 h2
      cases' h2 with _ _ _ _ _ hbc tbc _ _ y1 btc
      · cases h1
        exact SublistForall₂.nil
      · cases' h1 with _ _ _ _ _ hab tab _ _ _ atb
        · exact SublistForall₂.nil
        · exact SublistForall₂.cons (_root_.trans hab hbc) (ih _ _ tab tbc)
        · exact SublistForall₂.cons_right (ih _ _ atb tbc)
      · exact SublistForall₂.cons_right (ih _ _ h1 btc)⟩


theorem Sublist.sublistForall₂ {l₁ l₂ : List α} (h : l₁ <+ l₂) [IsRefl α Rₐ] :
    SublistForall₂ Rₐ l₁ l₂ :=
  sublistForall₂_iff.2 ⟨l₁, forall₂_refl l₁, h⟩


theorem tail_sublistForall₂_self [IsRefl α Rₐ] (l : List α) : SublistForall₂ Rₐ l.tail l :=
  l.tail_sublist.sublistForall₂


@[simp]
theorem sublistForall₂_map_left_iff {f : γ → α} {l₁ : List γ} {l₂ : List β} :
    SublistForall₂ R (map f l₁) l₂ ↔ SublistForall₂ (fun c b => R (f c) b) l₁ l₂ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    R : α → β → Prop
    f : γ → α
    l₁ : List γ
    l₂ : List β
    ⊢ Iff (List.SublistForall₂ R (List.map f l₁) l₂) (List.SublistForall₂ (fun c b …
  -/
  simp [sublistForall₂_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem sublistForall₂_map_right_iff {f : γ → β} {l₁ : List α} {l₂ : List γ} :
    SublistForall₂ R l₁ (map f l₂) ↔ SublistForall₂ (fun a c => R a (f c)) l₁ l₂ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    R : α → β → Prop
    f : γ → β
    l₁ : List α
    l₂ : List γ
    ⊢ Iff (List.SublistForall₂ R l₁ (List.map f l₂)) (List.SublistForall₂ (fun a c …
  -/
  simp only [sublistForall₂_iff]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    R : α → β → Prop
    f : γ → β
    l₁ : List α
    l₂ : List γ
    ⊢ Iff (Exists fun l => And (List.Forall₂ R l₁ l) (l.Sublist (List.map f l₂)))  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ : List γ
      ⊢ (Exists fun l => And (List.Forall₂ R l₁ l) (l.Sublist (List.map f l₂))) → Ex …
    -/
  · rintro ⟨l1, h1, h2⟩
    /-
      case mp.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ : List γ
      l1 : List β
      h1 : List.Forall₂ R l₁ l1
      h2 : l1.Sublist (List.map f l₂)
      ⊢ Exists fun l => And (List.Forall₂ (fun a c => R a (f c)) l₁ l) (l.Sublist l₂)
    -/
    obtain ⟨l', hl1, rfl⟩ := sublist_map_iff.mp h2
    /-
      case mp.intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ l' : List γ
      hl1 : l'.Sublist l₂
      h1 : List.Forall₂ R l₁ (List.map f l')
      h2 : (List.map f l').Sublist (List.map f l₂)
      ⊢ Exists fun l => And (List.Forall₂ (fun a c => R a (f c)) l₁ l) (l.Sublist l₂)
    -/
    use l'
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ l' : List γ
      hl1 : l'.Sublist l₂
      h1 : List.Forall₂ R l₁ (List.map f l')
      h2 : (List.map f l').Sublist (List.map f l₂)
      ⊢ And (List.Forall₂ (fun a c => R a (f c)) l₁ l') (l'.Sublist l₂)
    -/
    simpa [hl1] using h1
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ : List γ
      ⊢ (Exists fun l => And (List.Forall₂ (fun a c => R a (f c)) l₁ l) (l.Sublist l …
    -/
  · rintro ⟨l1, h1, h2⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ l1 : List γ
      h1 : List.Forall₂ (fun a c => R a (f c)) l₁ l1
      h2 : l1.Sublist l₂
      ⊢ Exists fun l => And (List.Forall₂ R l₁ l) (l.Sublist (List.map f l₂))
    -/
    use l1.map f
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      R : α → β → Prop
      f : γ → β
      l₁ : List α
      l₂ l1 : List γ
      h1 : List.Forall₂ (fun a c => R a (f c)) l₁ l1
      h2 : l1.Sublist l₂
      ⊢ And (List.Forall₂ R l₁ (List.map f l1)) ((List.map f l1).Sublist (List.map f …
    -/
    simp [h1, h2.map]
    /-
      🎉 no goals
    -/


