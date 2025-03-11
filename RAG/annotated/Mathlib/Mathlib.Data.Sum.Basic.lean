attribute [simp] Sum.forall Sum.exists


theorem exists_sum {γ : α ⊕ β → Sort*} (p : (∀ ab, γ ab) → Prop) :
    (∃ fab, p fab) ↔ (∃ fa fb, p (Sum.rec fa fb)) := by
  /-
    α : Type u
    β : Type v
    γ : Sum α β → Sort u_3
    p : ((ab : Sum α β) → γ ab) → Prop
    ⊢ Iff (Exists fun fab => p fab) (Exists fun fa => Exists fun fb => p fun t =>  …
  -/
  rw [← not_forall_not, forall_sum]
  /-
    α : Type u
    β : Type v
    γ : Sum α β → Sort u_3
    p : ((ab : Sum α β) → γ ab) → Prop
    ⊢ Iff (Not (∀ (fa : (val : α) → γ (Sum.inl val)) (fb : (val : β) → γ (Sum.inr  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem inl_injective : Function.Injective (inl : α → α ⊕ β) := fun _ _ ↦ inl.inj


theorem inr_injective : Function.Injective (inr : β → α ⊕ β) := fun _ _ ↦ inr.inj


theorem sum_rec_congr (P : α ⊕ β → Sort*) (f : ∀ i, P (inl i)) (g : ∀ i, P (inr i))
    {x y : α ⊕ β} (h : x = y) :
                                                                                  /-
                                                                                    α : Type u
                                                                                    β : Type v
                                                                                    P : Sum α β → Sort u_3
                                                                                    f : (i : α) → P (Sum.inl i)
                                                                                    g : (i : β) → P (Sum.inr i)
                                                                                    x y : Sum α β
                                                                                    h : Eq x y
                                                                                    ⊢ Eq (Sum.rec f g x) (cast ⋯ (Sum.rec f g y))
                                                                                  -/
    @Sum.rec _ _ _ f g x = cast (congr_arg P h.symm) (@Sum.rec _ _ _ f g y) := by cases h; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem eq_left_iff_getLeft_eq {a : α} : x = inl a ↔ ∃ h, x.getLeft h = a := by
  /-
    α : Type u
    β : Type v
    x : Sum α β
    a : α
    ⊢ Iff (Eq x (Sum.inl a)) (Exists fun h => Eq (x.getLeft h) a)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp
              /-
                🎉 no goals
              -/


theorem eq_right_iff_getRight_eq {b : β} : x = inr b ↔ ∃ h, x.getRight h = b := by
  /-
    α : Type u
    β : Type v
    x : Sum α β
    b : β
    ⊢ Iff (Eq x (Sum.inr b)) (Exists fun h => Eq (x.getRight h) b)
  -/
              /-
                🎉 no goals
              -/
  cases x <;> simp
              /-
                🎉 no goals
              -/


theorem getLeft_eq_getLeft? (h₁ : x.isLeft) (h₂ : x.getLeft?.isSome) :
                                           /-
                                             α : Type u
                                             β : Type v
                                             x : Sum α β
                                             h₁ : Eq x.isLeft Bool.true
                                             h₂ : Eq x.getLeft?.isSome Bool.true
                                             ⊢ Eq (x.getLeft h₁) (x.getLeft?.get h₂)
                                           -/
    x.getLeft h₁ = x.getLeft?.get h₂ := by simp [← getLeft?_eq_some_iff]
                                           /-
                                             🎉 no goals
                                           -/


theorem getRight_eq_getRight? (h₁ : x.isRight) (h₂ : x.getRight?.isSome) :
                                             /-
                                               α : Type u
                                               β : Type v
                                               x : Sum α β
                                               h₁ : Eq x.isRight Bool.true
                                               h₂ : Eq x.getRight?.isSome Bool.true
                                               ⊢ Eq (x.getRight h₁) (x.getRight?.get h₂)
                                             -/
    x.getRight h₁ = x.getRight?.get h₂ := by simp [← getRight?_eq_some_iff]
                                             /-
                                               🎉 no goals
                                             -/


@[simp] theorem isSome_getLeft?_iff_isLeft : x.getLeft?.isSome ↔ x.isLeft := by
  /-
    α : Type u
    β : Type v
    x : Sum α β
    ⊢ Iff (Eq x.getLeft?.isSome Bool.true) (Eq x.isLeft Bool.true)
  -/
  rw [isLeft_iff, Option.isSome_iff_exists]; simp
                                             /-
                                               🎉 no goals
                                             -/


@[simp] theorem isSome_getRight?_iff_isRight : x.getRight?.isSome ↔ x.isRight := by
  /-
    α : Type u
    β : Type v
    x : Sum α β
    ⊢ Iff (Eq x.getRight?.isSome Bool.true) (Eq x.isRight Bool.true)
  -/
  rw [isRight_iff, Option.isSome_iff_exists]; simp
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem update_elim_inl [DecidableEq α] [DecidableEq (α ⊕ β)] {f : α → γ} {g : β → γ} {i : α}
    {x : γ} : update (Sum.elim f g) (inl i) x = Sum.elim (update f i x) g :=
                      /-
                        α : Type u
                        β : Type v
                        γ : Type u_1
                        inst✝¹ : DecidableEq α
                        inst✝ : DecidableEq (Sum α β)
                        f : α → γ
                        g : β → γ
                        i : α
                        x : γ
                        ⊢ Eq x (Sum.elim (Function.update f i x) g (Sum.inl i))
                      -/
                      /-
                        🎉 no goals
                      -/
  update_eq_iff.2 ⟨by simp, by simp +contextual⟩
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem update_elim_inr [DecidableEq β] [DecidableEq (α ⊕ β)] {f : α → γ} {g : β → γ} {i : β}
    {x : γ} : update (Sum.elim f g) (inr i) x = Sum.elim f (update g i x) :=
                      /-
                        α : Type u
                        β : Type v
                        γ : Type u_1
                        inst✝¹ : DecidableEq β
                        inst✝ : DecidableEq (Sum α β)
                        f : α → γ
                        g : β → γ
                        i : β
                        x : γ
                        ⊢ Eq x (Sum.elim f (Function.update g i x) (Sum.inr i))
                      -/
                      /-
                        🎉 no goals
                      -/
  update_eq_iff.2 ⟨by simp, by simp +contextual⟩
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem update_inl_comp_inl [DecidableEq α] [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i : α}
    {x : γ} : update f (inl i) x ∘ inl = update (f ∘ inl) i x :=
  update_comp_eq_of_injective _ inl_injective _ _


@[simp]
theorem update_inl_apply_inl [DecidableEq α] [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i j : α}
    {x : γ} : update f (inl i) x (inl j) = update (f ∘ inl) i x j := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq (Sum α β)
    f : Sum α β → γ
    i j : α
    x : γ
    ⊢ Eq (Function.update f (Sum.inl i) x (Sum.inl j)) (Function.update (Function. …
  -/
  rw [← update_inl_comp_inl, Function.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem update_inl_comp_inr [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i : α} {x : γ} :
    update f (inl i) x ∘ inr = f ∘ inr :=
  (update_comp_eq_of_forall_ne _ _) fun _ ↦ inr_ne_inl


theorem update_inl_apply_inr [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i : α} {j : β} {x : γ} :
    update f (inl i) x (inr j) = f (inr j) :=
  Function.update_of_ne inr_ne_inl ..


@[simp]
theorem update_inr_comp_inl [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i : β} {x : γ} :
    update f (inr i) x ∘ inl = f ∘ inl :=
  (update_comp_eq_of_forall_ne _ _) fun _ ↦ inl_ne_inr


theorem update_inr_apply_inl [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i : α} {j : β} {x : γ} :
    update f (inr j) x (inl i) = f (inl i) :=
  Function.update_of_ne inl_ne_inr ..


@[simp]
theorem update_inr_comp_inr [DecidableEq β] [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i : β}
    {x : γ} : update f (inr i) x ∘ inr = update (f ∘ inr) i x :=
  update_comp_eq_of_injective _ inr_injective _ _


@[simp]
theorem update_inr_apply_inr [DecidableEq β] [DecidableEq (α ⊕ β)] {f : α ⊕ β → γ} {i j : β}
    {x : γ} : update f (inr i) x (inr j) = update (f ∘ inr) i x j := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq (Sum α β)
    f : Sum α β → γ
    i j : β
    x : γ
    ⊢ Eq (Function.update f (Sum.inr i) x (Sum.inr j)) (Function.update (Function. …
  -/
  rw [← update_inr_comp_inr, Function.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem swap_leftInverse : Function.LeftInverse (@swap α β) swap :=
  swap_swap


@[simp]
theorem swap_rightInverse : Function.RightInverse (@swap α β) swap :=
  swap_swap


mk_iff_of_inductive_prop Sum.LiftRel Sum.liftRel_iff


                                                                       /-
                                                                         α : Type u
                                                                         β : Type v
                                                                         γ : Type u_1
                                                                         δ : Type u_2
                                                                         r : α → γ → Prop
                                                                         s : β → δ → Prop
                                                                         x : Sum α β
                                                                         y : Sum γ δ
                                                                         h : Sum.LiftRel r s x y
                                                                         ⊢ Iff (Eq x.isLeft Bool.true) (Eq y.isLeft Bool.true)
                                                                       -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
theorem isLeft_congr (h : LiftRel r s x y) : x.isLeft ↔ y.isLeft := by cases h <;> rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/

                                                                          /-
                                                                            α : Type u
                                                                            β : Type v
                                                                            γ : Type u_1
                                                                            δ : Type u_2
                                                                            r : α → γ → Prop
                                                                            s : β → δ → Prop
                                                                            x : Sum α β
                                                                            y : Sum γ δ
                                                                            h : Sum.LiftRel r s x y
                                                                            ⊢ Iff (Eq x.isRight Bool.true) (Eq y.isRight Bool.true)
                                                                          -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
theorem isRight_congr (h : LiftRel r s x y) : x.isRight ↔ y.isRight := by cases h <;> rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


                                                                 /-
                                                                   α : Type u
                                                                   β : Type v
                                                                   γ : Type u_1
                                                                   δ : Type u_2
                                                                   r : α → γ → Prop
                                                                   s : β → δ → Prop
                                                                   x : Sum α β
                                                                   c : γ
                                                                   h : Sum.LiftRel r s x (Sum.inl c)
                                                                   ⊢ Eq x.isLeft Bool.true
                                                                 -/
theorem isLeft_left (h : LiftRel r s x (inl c)) : x.isLeft := by cases h; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

                                                                  /-
                                                                    α : Type u
                                                                    β : Type v
                                                                    γ : Type u_1
                                                                    δ : Type u_2
                                                                    r : α → γ → Prop
                                                                    s : β → δ → Prop
                                                                    y : Sum γ δ
                                                                    a : α
                                                                    h : Sum.LiftRel r s (Sum.inl a) y
                                                                    ⊢ Eq y.isLeft Bool.true
                                                                  -/
theorem isLeft_right (h : LiftRel r s (inl a) y) : y.isLeft := by cases h; rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/

                                                                   /-
                                                                     α : Type u
                                                                     β : Type v
                                                                     γ : Type u_1
                                                                     δ : Type u_2
                                                                     r : α → γ → Prop
                                                                     s : β → δ → Prop
                                                                     x : Sum α β
                                                                     d : δ
                                                                     h : Sum.LiftRel r s x (Sum.inr d)
                                                                     ⊢ Eq x.isRight Bool.true
                                                                   -/
theorem isRight_left (h : LiftRel r s x (inr d)) : x.isRight := by cases h; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/

                                                                    /-
                                                                      α : Type u
                                                                      β : Type v
                                                                      γ : Type u_1
                                                                      δ : Type u_2
                                                                      r : α → γ → Prop
                                                                      s : β → δ → Prop
                                                                      y : Sum γ δ
                                                                      b : β
                                                                      h : Sum.LiftRel r s (Sum.inr b) y
                                                                      ⊢ Eq y.isRight Bool.true
                                                                    -/
theorem isRight_right (h : LiftRel r s (inr b) y) : y.isRight := by cases h; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem exists_of_isLeft_left (h₁ : LiftRel r s x y) (h₂ : x.isLeft) :
    ∃ a c, r a c ∧ x = inl a ∧ y = inl c := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    δ : Type u_2
    r : α → γ → Prop
    s : β → δ → Prop
    x : Sum α β
    y : Sum γ δ
    h₁ : Sum.LiftRel r s x y
    h₂ : Eq x.isLeft Bool.true
    ⊢ Exists fun a => Exists fun c => And (r a c) (And (Eq x (Sum.inl a)) (Eq y (S …
  -/
  rcases isLeft_iff.mp h₂ with ⟨_, rfl⟩
  /-
    case intro
    α : Type u
    β : Type v
    γ : Type u_1
    δ : Type u_2
    r : α → γ → Prop
    s : β → δ → Prop
    y : Sum γ δ
    w✝ : α
    h₁ : Sum.LiftRel r s (Sum.inl w✝) y
    h₂ : Eq (Sum.inl w✝).isLeft Bool.true
    ⊢ Exists fun a => Exists fun c => And (r a c) (And (Eq (Sum.inl w✝) (Sum.inl a …
  -/
  simp only [liftRel_iff, false_and, and_false, exists_false, or_false, reduceCtorEq] at h₁
  /-
    case intro
    α : Type u
    β : Type v
    γ : Type u_1
    δ : Type u_2
    r : α → γ → Prop
    s : β → δ → Prop
    y : Sum γ δ
    w✝ : α
    h₂ : Eq (Sum.inl w✝).isLeft Bool.true
    h₁ : Exists fun a => Exists fun c => And (r a c) (And (Eq (Sum.inl w✝) (Sum.in …
    ⊢ Exists fun a => Exists fun c => And (r a c) (And (Eq (Sum.inl w✝) (Sum.inl a …
  -/
  exact h₁
  /-
    🎉 no goals
  -/


theorem exists_of_isLeft_right (h₁ : LiftRel r s x y) (h₂ : y.isLeft) :
    ∃ a c, r a c ∧ x = inl a ∧ y = inl c := exists_of_isLeft_left h₁ ((isLeft_congr h₁).mpr h₂)


theorem exists_of_isRight_left (h₁ : LiftRel r s x y) (h₂ : x.isRight) :
    ∃ b d, s b d ∧ x = inr b ∧ y = inr d := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    δ : Type u_2
    r : α → γ → Prop
    s : β → δ → Prop
    x : Sum α β
    y : Sum γ δ
    h₁ : Sum.LiftRel r s x y
    h₂ : Eq x.isRight Bool.true
    ⊢ Exists fun b => Exists fun d => And (s b d) (And (Eq x (Sum.inr b)) (Eq y (S …
  -/
  rcases isRight_iff.mp h₂ with ⟨_, rfl⟩
  /-
    case intro
    α : Type u
    β : Type v
    γ : Type u_1
    δ : Type u_2
    r : α → γ → Prop
    s : β → δ → Prop
    y : Sum γ δ
    w✝ : β
    h₁ : Sum.LiftRel r s (Sum.inr w✝) y
    h₂ : Eq (Sum.inr w✝).isRight Bool.true
    ⊢ Exists fun b => Exists fun d => And (s b d) (And (Eq (Sum.inr w✝) (Sum.inr b …
  -/
  simp only [liftRel_iff, false_and, and_false, exists_false, false_or, reduceCtorEq] at h₁
  /-
    case intro
    α : Type u
    β : Type v
    γ : Type u_1
    δ : Type u_2
    r : α → γ → Prop
    s : β → δ → Prop
    y : Sum γ δ
    w✝ : β
    h₂ : Eq (Sum.inr w✝).isRight Bool.true
    h₁ : Exists fun b => Exists fun d => And (s b d) (And (Eq (Sum.inr w✝) (Sum.in …
    ⊢ Exists fun b => Exists fun d => And (s b d) (And (Eq (Sum.inr w✝) (Sum.inr b …
  -/
  exact h₁
  /-
    🎉 no goals
  -/


theorem exists_of_isRight_right (h₁ : LiftRel r s x y) (h₂ : y.isRight) :
    ∃ b d, s b d ∧ x = inr b ∧ y = inr d :=
  exists_of_isRight_left h₁ ((isRight_congr h₁).mpr h₂)


theorem Injective.sum_elim {f : α → γ} {g : β → γ} (hf : Injective f) (hg : Injective g)
    (hfg : ∀ a b, f a ≠ g b) : Injective (Sum.elim f g)
  | inl _, inl _, h => congr_arg inl <| hf h
  | inl _, inr _, h => (hfg _ _ h).elim
  | inr _, inl _, h => (hfg _ _ h.symm).elim
  | inr _, inr _, h => congr_arg inr <| hg h


theorem Injective.sum_map {f : α → β} {g : α' → β'} (hf : Injective f) (hg : Injective g) :
    Injective (Sum.map f g)
  | inl _, inl _, h => congr_arg inl <| hf <| inl.inj h
  | inr _, inr _, h => congr_arg inr <| hg <| inr.inj h


theorem Surjective.sum_map {f : α → β} {g : α' → β'} (hf : Surjective f) (hg : Surjective g) :
    Surjective (Sum.map f g)
  | inl y =>
    let ⟨x, hx⟩ := hf y
    ⟨inl x, congr_arg inl hx⟩
  | inr y =>
    let ⟨x, hx⟩ := hg y
    ⟨inr x, congr_arg inr hx⟩


theorem Bijective.sum_map {f : α → β} {g : α' → β'} (hf : Bijective f) (hg : Bijective g) :
    Bijective (Sum.map f g) :=
  ⟨hf.injective.sum_map hg.injective, hf.surjective.sum_map hg.surjective⟩


@[simp]
theorem map_injective {f : α → γ} {g : β → δ} :
    Injective (Sum.map f g) ↔ Injective f ∧ Injective g :=
  ⟨fun h =>
    ⟨fun a₁ a₂ ha => inl_injective <| @h (inl a₁) (inl a₂) (congr_arg inl ha : _), fun b₁ b₂ hb =>
      inr_injective <| @h (inr b₁) (inr b₂) (congr_arg inr hb : _)⟩,
    fun h => h.1.sum_map h.2⟩


@[simp]
theorem map_surjective {f : α → γ} {g : β → δ} :
    Surjective (Sum.map f g) ↔ Surjective f ∧ Surjective g :=
  ⟨ fun h => ⟨
      (fun c => by
        /-
          α : Type u
          β : Type v
          γ : Type u_1
          δ : Type u_2
          f : α → γ
          g : β → δ
          h : Function.Surjective (Sum.map f g)
          c : γ
          ⊢ Exists fun a => Eq (f a) c
        -/
        obtain ⟨a | b, h⟩ := h (inl c)
          /-
            case intro.inl
            α : Type u
            β : Type v
            γ : Type u_1
            δ : Type u_2
            f : α → γ
            g : β → δ
            h✝ : Function.Surjective (Sum.map f g)
            c : γ
            a : α
            h : Eq (Sum.map f g (Sum.inl a)) (Sum.inl c)
            ⊢ Exists fun a => Eq (f a) c
          -/
        · exact ⟨a, inl_injective h⟩
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u
            β : Type v
            γ : Type u_1
            δ : Type u_2
            f : α → γ
            g : β → δ
            h✝ : Function.Surjective (Sum.map f g)
            c : γ
            b : β
            h : Eq (Sum.map f g (Sum.inr b)) (Sum.inl c)
            ⊢ Exists fun a => Eq (f a) c
          -/
        · cases h),
          /-
            🎉 no goals
          -/
      (fun d => by
        /-
          α : Type u
          β : Type v
          γ : Type u_1
          δ : Type u_2
          f : α → γ
          g : β → δ
          h : Function.Surjective (Sum.map f g)
          d : δ
          ⊢ Exists fun a => Eq (g a) d
        -/
        obtain ⟨a | b, h⟩ := h (inr d)
          /-
            case intro.inl
            α : Type u
            β : Type v
            γ : Type u_1
            δ : Type u_2
            f : α → γ
            g : β → δ
            h✝ : Function.Surjective (Sum.map f g)
            d : δ
            a : α
            h : Eq (Sum.map f g (Sum.inl a)) (Sum.inr d)
            ⊢ Exists fun a => Eq (g a) d
          -/
        · cases h
          /-
            🎉 no goals
          -/
          /-
            case intro.inr
            α : Type u
            β : Type v
            γ : Type u_1
            δ : Type u_2
            f : α → γ
            g : β → δ
            h✝ : Function.Surjective (Sum.map f g)
            d : δ
            b : β
            h : Eq (Sum.map f g (Sum.inr b)) (Sum.inr d)
            ⊢ Exists fun a => Eq (g a) d
          -/
        · exact ⟨b, inr_injective h⟩)⟩,
          /-
            🎉 no goals
          -/
    fun h => h.1.sum_map h.2⟩


@[simp]
theorem map_bijective {f : α → γ} {g : β → δ} :
    Bijective (Sum.map f g) ↔ Bijective f ∧ Bijective g :=
  (map_injective.and map_surjective).trans <| and_and_and_comm


theorem elim_update_left [DecidableEq α] [DecidableEq β] (f : α → γ) (g : β → γ) (i : α) (c : γ) :
    Sum.elim (Function.update f i c) g = Function.update (Sum.elim f g) (inl i) c := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → γ
    g : β → γ
    i : α
    c : γ
    ⊢ Eq (Sum.elim (Function.update f i c) g) (Function.update (Sum.elim f g) (Sum …
  -/
  ext x
  /-
    case h
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → γ
    g : β → γ
    i : α
    c : γ
    x : Sum α β
    ⊢ Eq (Sum.elim (Function.update f i c) g x) (Function.update (Sum.elim f g) (S …
  -/
  rcases x with x | x
    /-
      case h.inl
      α : Type u
      β : Type v
      γ : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → γ
      g : β → γ
      i : α
      c : γ
      x : α
      ⊢ Eq (Sum.elim (Function.update f i c) g (Sum.inl x)) (Function.update (Sum.el …
    -/
  · by_cases h : x = i
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → γ
        g : β → γ
        i : α
        c : γ
        x : α
        h : Eq x i
        ⊢ Eq (Sum.elim (Function.update f i c) g (Sum.inl x)) (Function.update (Sum.el …
      -/
    · subst h
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → γ
        g : β → γ
        c : γ
        x : α
        ⊢ Eq (Sum.elim (Function.update f x c) g (Sum.inl x)) (Function.update (Sum.el …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → γ
        g : β → γ
        i : α
        c : γ
        x : α
        h : Not (Eq x i)
        ⊢ Eq (Sum.elim (Function.update f i c) g (Sum.inl x)) (Function.update (Sum.el …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      α : Type u
      β : Type v
      γ : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → γ
      g : β → γ
      i : α
      c : γ
      x : β
      ⊢ Eq (Sum.elim (Function.update f i c) g (Sum.inr x)) (Function.update (Sum.el …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem elim_update_right [DecidableEq α] [DecidableEq β] (f : α → γ) (g : β → γ) (i : β) (c : γ) :
    Sum.elim f (Function.update g i c) = Function.update (Sum.elim f g) (inr i) c := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → γ
    g : β → γ
    i : β
    c : γ
    ⊢ Eq (Sum.elim f (Function.update g i c)) (Function.update (Sum.elim f g) (Sum …
  -/
  ext x
  /-
    case h
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → γ
    g : β → γ
    i : β
    c : γ
    x : Sum α β
    ⊢ Eq (Sum.elim f (Function.update g i c) x) (Function.update (Sum.elim f g) (S …
  -/
  rcases x with x | x
    /-
      case h.inl
      α : Type u
      β : Type v
      γ : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → γ
      g : β → γ
      i : β
      c : γ
      x : α
      ⊢ Eq (Sum.elim f (Function.update g i c) (Sum.inl x)) (Function.update (Sum.el …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u
      β : Type v
      γ : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → γ
      g : β → γ
      i : β
      c : γ
      x : β
      ⊢ Eq (Sum.elim f (Function.update g i c) (Sum.inr x)) (Function.update (Sum.el …
    -/
  · by_cases h : x = i
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → γ
        g : β → γ
        i : β
        c : γ
        x : β
        h : Eq x i
        ⊢ Eq (Sum.elim f (Function.update g i c) (Sum.inr x)) (Function.update (Sum.el …
      -/
    · subst h
      /-
        case pos
        α : Type u
        β : Type v
        γ : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → γ
        g : β → γ
        c : γ
        x : β
        ⊢ Eq (Sum.elim f (Function.update g x c) (Sum.inr x)) (Function.update (Sum.el …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        γ : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → γ
        g : β → γ
        i : β
        c : γ
        x : β
        h : Not (Eq x i)
        ⊢ Eq (Sum.elim f (Function.update g i c) (Sum.inr x)) (Function.update (Sum.el …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/


/-- The map from the first summand into a ternary sum. -/
@[match_pattern, simp, reducible]
def in₀ (a : α) : α ⊕ (β ⊕ γ) :=
  inl a


/-- The map from the second summand into a ternary sum. -/
@[match_pattern, simp, reducible]
def in₁ (b : β) : α ⊕ (β ⊕ γ) :=
  inr <| inl b


/-- The map from the third summand into a ternary sum. -/
@[match_pattern, simp, reducible]
def in₂ (c : γ) : α ⊕ (β ⊕ γ) :=
  inr <| inr c


