/-- Predicate for semiring homomorphisms (deprecated -- use the bundled `RingHom` version). -/
structure IsSemiringHom {α : Type u} {β : Type v} [Semiring α] [Semiring β] (f : α → β) : Prop where
  /-- The proposition that `f` preserves the additive identity. -/
  map_zero : f 0 = 0
  /-- The proposition that `f` preserves the multiplicative identity. -/
  map_one : f 1 = 1
  /-- The proposition that `f` preserves addition. -/
  map_add : ∀ x y, f (x + y) = f x + f y
  /-- The proposition that `f` preserves multiplication. -/
  map_mul : ∀ x y, f (x * y) = f x * f y


/-- The identity map is a semiring homomorphism. -/
                                         /-
                                           α : Type u
                                           inst✝ : Semiring α
                                           ⊢ IsSemiringHom _root_.id
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
theorem id : IsSemiringHom (@id α) := by constructor <;> intros <;> rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The composition of two semiring homomorphisms is a semiring homomorphism. -/
theorem comp (hf : IsSemiringHom f) {γ} [Semiring γ] {g : β → γ} (hg : IsSemiringHom g) :
    IsSemiringHom (g ∘ f) :=
                   /-
                     α : Type u
                     β : Type v
                     inst✝² : Semiring α
                     inst✝¹ : Semiring β
                     f : α → β
                     hf : IsSemiringHom f
                     γ : Type u_1
                     inst✝ : Semiring γ
                     g : β → γ
                     hg : IsSemiringHom g
                     ⊢ Eq (Function.comp g f 0) 0
                   -/
  { map_zero := by simpa [map_zero hf] using map_zero hg
                   /-
                     🎉 no goals
                   -/
                  /-
                    α : Type u
                    β : Type v
                    inst✝² : Semiring α
                    inst✝¹ : Semiring β
                    f : α → β
                    hf : IsSemiringHom f
                    γ : Type u_1
                    inst✝ : Semiring γ
                    g : β → γ
                    hg : IsSemiringHom g
                    ⊢ Eq (Function.comp g f 1) 1
                  -/
    map_one := by simpa [map_one hf] using map_one hg
                  /-
                    🎉 no goals
                  -/
                               /-
                                 α : Type u
                                 β : Type v
                                 inst✝² : Semiring α
                                 inst✝¹ : Semiring β
                                 f : α → β
                                 hf : IsSemiringHom f
                                 γ : Type u_1
                                 inst✝ : Semiring γ
                                 g : β → γ
                                 hg : IsSemiringHom g
                                 x y : α
                                 ⊢ Eq (Function.comp g f (HAdd.hAdd x y)) (HAdd.hAdd (Function.comp g f x) (Fun …
                               -/
    map_add := fun {x y} => by simp [map_add hf, map_add hg]
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 α : Type u
                                 β : Type v
                                 inst✝² : Semiring α
                                 inst✝¹ : Semiring β
                                 f : α → β
                                 hf : IsSemiringHom f
                                 γ : Type u_1
                                 inst✝ : Semiring γ
                                 g : β → γ
                                 hg : IsSemiringHom g
                                 x y : α
                                 ⊢ Eq (Function.comp g f (HMul.hMul x y)) (HMul.hMul (Function.comp g f x) (Fun …
                               -/
    map_mul := fun {x y} => by simp [map_mul hf, map_mul hg] }
                               /-
                                 🎉 no goals
                               -/


/-- A semiring homomorphism is an additive monoid homomorphism. -/
theorem to_isAddMonoidHom (hf : IsSemiringHom f) : IsAddMonoidHom f :=
                                         /-
                                           α : Type u
                                           β : Type v
                                           inst✝¹ : Semiring α
                                           inst✝ : Semiring β
                                           f : α → β
                                           hf : IsSemiringHom f
                                           ⊢ ∀ (x y : α), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                                         -/
  { ‹IsSemiringHom f› with map_add := by apply @‹IsSemiringHom f›.map_add }
                                         /-
                                           🎉 no goals
                                         -/


/-- A semiring homomorphism is a monoid homomorphism. -/
theorem to_isMonoidHom (hf : IsSemiringHom f) : IsMonoidHom f :=
  { ‹IsSemiringHom f› with }


/-- Predicate for ring homomorphisms (deprecated -- use the bundled `RingHom` version). -/
structure IsRingHom {α : Type u} {β : Type v} [Ring α] [Ring β] (f : α → β) : Prop where
  /-- The proposition that `f` preserves the multiplicative identity. -/
  map_one : f 1 = 1
  /-- The proposition that `f` preserves multiplication. -/
  map_mul : ∀ x y, f (x * y) = f x * f y
  /-- The proposition that `f` preserves addition. -/
  map_add : ∀ x y, f (x + y) = f x + f y


/-- A map of rings that is a semiring homomorphism is also a ring homomorphism. -/
theorem of_semiring {f : α → β} (H : IsSemiringHom f) : IsRingHom f :=
  { H with }


/-- Ring homomorphisms map zero to zero. -/
theorem map_zero (hf : IsRingHom f) : f 0 = 0 :=
  calc
                                /-
                                  α : Type u
                                  β : Type v
                                  inst✝¹ : Ring α
                                  inst✝ : Ring β
                                  f : α → β
                                  hf : IsRingHom f
                                  ⊢ Eq (f 0) (HSub.hSub (f (HAdd.hAdd 0 0)) (f 0))
                                -/
    f 0 = f (0 + 0) - f 0 := by rw [hf.map_add]; simp
                                                 /-
                                                   🎉 no goals
                                                 -/
                /-
                  α : Type u
                  β : Type v
                  inst✝¹ : Ring α
                  inst✝ : Ring β
                  f : α → β
                  hf : IsRingHom f
                  ⊢ Eq (HSub.hSub (f (HAdd.hAdd 0 0)) (f 0)) 0
                -/
    _ = 0 := by simp
                /-
                  🎉 no goals
                -/


/-- Ring homomorphisms preserve additive inverses. -/
theorem map_neg (hf : IsRingHom f) : f (-x) = -f x :=
  calc
                                    /-
                                      α : Type u
                                      β : Type v
                                      inst✝¹ : Ring α
                                      inst✝ : Ring β
                                      f : α → β
                                      x : α
                                      hf : IsRingHom f
                                      ⊢ Eq (f (Neg.neg x)) (HSub.hSub (f (HAdd.hAdd (Neg.neg x) x)) (f x))
                                    -/
    f (-x) = f (-x + x) - f x := by rw [hf.map_add]; simp
                                                     /-
                                                       🎉 no goals
                                                     -/
                   /-
                     α : Type u
                     β : Type v
                     inst✝¹ : Ring α
                     inst✝ : Ring β
                     f : α → β
                     x : α
                     hf : IsRingHom f
                     ⊢ Eq (HSub.hSub (f (HAdd.hAdd (Neg.neg x) x)) (f x)) (Neg.neg (f x))
                   -/
    _ = -f x := by simp [hf.map_zero]
                   /-
                     🎉 no goals
                   -/


/-- Ring homomorphisms preserve subtraction. -/
theorem map_sub (hf : IsRingHom f) : f (x - y) = f x - f y := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Ring α
    inst✝ : Ring β
    f : α → β
    x y : α
    hf : IsRingHom f
    ⊢ Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
  -/
  simp [sub_eq_add_neg, hf.map_add, hf.map_neg]
  /-
    🎉 no goals
  -/


/-- The identity map is a ring homomorphism. -/
                                     /-
                                       α : Type u
                                       inst✝ : Ring α
                                       ⊢ IsRingHom _root_.id
                                     -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem id : IsRingHom (@id α) := by constructor <;> intros <;> rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- see Note [no instance on morphisms]

/-- The composition of two ring homomorphisms is a ring homomorphism. -/
theorem comp (hf : IsRingHom f) {γ} [Ring γ] {g : β → γ} (hg : IsRingHom g) : IsRingHom (g ∘ f) :=
                             /-
                               α : Type u
                               β : Type v
                               inst✝² : Ring α
                               inst✝¹ : Ring β
                               f : α → β
                               hf : IsRingHom f
                               γ : Type u_1
                               inst✝ : Ring γ
                               g : β → γ
                               hg : IsRingHom g
                               x y : α
                               ⊢ Eq (Function.comp g f (HAdd.hAdd x y)) (HAdd.hAdd (Function.comp g f x) (Fun …
                             -/
                             /-
                               α : Type u
                               β : Type v
                               inst✝² : Ring α
                               inst✝¹ : Ring β
                               f : α → β
                               hf : IsRingHom f
                               γ : Type u_1
                               inst✝ : Ring γ
                               g : β → γ
                               hg : IsRingHom g
                               x y : α
                               ⊢ Eq (Function.comp g f (HMul.hMul x y)) (HMul.hMul (Function.comp g f x) (Fun …
                             -/
                  /-
                    α : Type u
                    β : Type v
                    inst✝² : Ring α
                    inst✝¹ : Ring β
                    f : α → β
                    hf : IsRingHom f
                    γ : Type u_1
                    inst✝ : Ring γ
                    g : β → γ
                    hg : IsRingHom g
                    ⊢ Eq (Function.comp g f 1) 1
                  -/
  { map_add := fun x y => by simp only [Function.comp_apply, map_add hf, map_add hg]
                  /-
                    🎉 no goals
                  -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
    map_mul := fun x y => by simp only [Function.comp_apply, map_mul hf, map_mul hg]
    map_one := by simp only [Function.comp_apply, map_one hf, map_one hg] }


/-- A ring homomorphism is also a semiring homomorphism. -/
theorem to_isSemiringHom (hf : IsRingHom f) : IsSemiringHom f :=
  { ‹IsRingHom f› with map_zero := map_zero hf }


theorem to_isAddGroupHom (hf : IsRingHom f) : IsAddGroupHom f :=
  { map_add := hf.map_add }


/-- Interpret `f : α → β` with `IsSemiringHom f` as a ring homomorphism. -/
def of {f : α → β} (hf : IsSemiringHom f) : α →+* β :=
  { MonoidHom.of hf.to_isMonoidHom, AddMonoidHom.of hf.to_isAddMonoidHom with toFun := f }


@[simp]
theorem coe_of {f : α → β} (hf : IsSemiringHom f) : ⇑(of hf) = f :=
  rfl


theorem to_isSemiringHom (f : α →+* β) : IsSemiringHom f :=
  { map_zero := f.map_zero
    map_one := f.map_one
    map_add := f.map_add
    map_mul := f.map_mul }


theorem to_isRingHom {α γ} [Ring α] [Ring γ] (g : α →+* γ) : IsRingHom g :=
  IsRingHom.of_semiring g.to_isSemiringHom


