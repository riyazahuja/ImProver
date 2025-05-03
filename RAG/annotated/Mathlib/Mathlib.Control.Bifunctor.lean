/-- Lawless bifunctor. This typeclass only holds the data for the bimap. -/
class Bifunctor (F : Type u₀ → Type u₁ → Type u₂) where
  bimap : ∀ {α α' β β'}, (α → α') → (β → β') → F α β → F α' β'


/-- Bifunctor. This typeclass asserts that a lawless `Bifunctor` is lawful. -/
class LawfulBifunctor (F : Type u₀ → Type u₁ → Type u₂) [Bifunctor F] : Prop where
  id_bimap : ∀ {α β} (x : F α β), bimap id id x = x
  bimap_bimap :
    ∀ {α₀ α₁ α₂ β₀ β₁ β₂} (f : α₀ → α₁) (f' : α₁ → α₂) (g : β₀ → β₁) (g' : β₁ → β₂) (x : F α₀ β₀),
      bimap f' g' (bimap f g x) = bimap (f' ∘ f) (g' ∘ g) x


attribute [higher_order bimap_id_id] id_bimap


attribute [higher_order bimap_comp_bimap] bimap_bimap


/-- Left map of a bifunctor. -/
abbrev fst {α α' β} (f : α → α') : F α β → F α' β :=
  bimap f id


/-- Right map of a bifunctor. -/
abbrev snd {α β β'} (f : β → β') : F α β → F α β' :=
  bimap id f


@[higher_order fst_id]
theorem id_fst : ∀ {α β} (x : F α β), fst id x = x :=
  @id_bimap _ _ _


@[higher_order snd_id]
theorem id_snd : ∀ {α β} (x : F α β), snd id x = x :=
  @id_bimap _ _ _


@[higher_order fst_comp_fst]
theorem comp_fst {α₀ α₁ α₂ β} (f : α₀ → α₁) (f' : α₁ → α₂) (x : F α₀ β) :
                                            /-
                                              F : Type u₀ → Type u₁ → Type u₂
                                              inst✝¹ : Bifunctor F
                                              inst✝ : LawfulBifunctor F
                                              α₀ α₁ α₂ : Type u₀
                                              β : Type u₁
                                              f : α₀ → α₁
                                              f' : α₁ → α₂
                                              x : F α₀ β
                                              ⊢ Eq (Bifunctor.fst f' (Bifunctor.fst f x)) (Bifunctor.fst (Function.comp f' f …
                                            -/
    fst f' (fst f x) = fst (f' ∘ f) x := by simp [fst, bimap_bimap]
                                            /-
                                              🎉 no goals
                                            -/


@[higher_order fst_comp_snd]
theorem fst_snd {α₀ α₁ β₀ β₁} (f : α₀ → α₁) (f' : β₀ → β₁) (x : F α₀ β₀) :
                                          /-
                                            F : Type u₀ → Type u₁ → Type u₂
                                            inst✝¹ : Bifunctor F
                                            inst✝ : LawfulBifunctor F
                                            α₀ α₁ : Type u₀
                                            β₀ β₁ : Type u₁
                                            f : α₀ → α₁
                                            f' : β₀ → β₁
                                            x : F α₀ β₀
                                            ⊢ Eq (Bifunctor.fst f (Bifunctor.snd f' x)) (Bifunctor.bimap f f' x)
                                          -/
    fst f (snd f' x) = bimap f f' x := by simp [fst, bimap_bimap]
                                          /-
                                            🎉 no goals
                                          -/


@[higher_order snd_comp_fst]
theorem snd_fst {α₀ α₁ β₀ β₁} (f : α₀ → α₁) (f' : β₀ → β₁) (x : F α₀ β₀) :
                                          /-
                                            F : Type u₀ → Type u₁ → Type u₂
                                            inst✝¹ : Bifunctor F
                                            inst✝ : LawfulBifunctor F
                                            α₀ α₁ : Type u₀
                                            β₀ β₁ : Type u₁
                                            f : α₀ → α₁
                                            f' : β₀ → β₁
                                            x : F α₀ β₀
                                            ⊢ Eq (Bifunctor.snd f' (Bifunctor.fst f x)) (Bifunctor.bimap f f' x)
                                          -/
    snd f' (fst f x) = bimap f f' x := by simp [snd, bimap_bimap]
                                          /-
                                            🎉 no goals
                                          -/


@[higher_order snd_comp_snd]
theorem comp_snd {α β₀ β₁ β₂} (g : β₀ → β₁) (g' : β₁ → β₂) (x : F α β₀) :
                                            /-
                                              F : Type u₀ → Type u₁ → Type u₂
                                              inst✝¹ : Bifunctor F
                                              inst✝ : LawfulBifunctor F
                                              α : Type u₀
                                              β₀ β₁ β₂ : Type u₁
                                              g : β₀ → β₁
                                              g' : β₁ → β₂
                                              x : F α β₀
                                              ⊢ Eq (Bifunctor.snd g' (Bifunctor.snd g x)) (Bifunctor.snd (Function.comp g' g …
                                            -/
    snd g' (snd g x) = snd (g' ∘ g) x := by simp [snd, bimap_bimap]
                                            /-
                                              🎉 no goals
                                            -/


instance Prod.bifunctor : Bifunctor Prod where bimap := @Prod.map


instance Prod.lawfulBifunctor : LawfulBifunctor Prod where
  id_bimap _ := rfl
  bimap_bimap _ _ _ _ _ := rfl


instance Bifunctor.const : Bifunctor Const where bimap f _ := f


instance LawfulBifunctor.const : LawfulBifunctor Const where
  id_bimap _ := rfl
  bimap_bimap _ _ _ _ _ := rfl


instance Bifunctor.flip : Bifunctor (flip F) where
  bimap {_α α' _β β'} f f' x := (bimap f' f x : F β' α')


instance LawfulBifunctor.flip [LawfulBifunctor F] : LawfulBifunctor (flip F) where
                 /-
                   F : Type u₀ → Type u₁ → Type u₂
                   inst✝¹ : Bifunctor F
                   inst✝ : LawfulBifunctor F
                   ⊢ ∀ {α : Type u₁} {β : Type u₀} (x : _root_.flip F α β), Eq (Bifunctor.bimap i …
                 -/
  id_bimap := by simp [bimap, functor_norm]
                 /-
                   🎉 no goals
                 -/
                    /-
                      F : Type u₀ → Type u₁ → Type u₂
                      inst✝¹ : Bifunctor F
                      inst✝ : LawfulBifunctor F
                      ⊢ ∀ {α₀ α₁ α₂ : Type u₁} {β₀ β₁ β₂ : Type u₀} (f : α₀ → α₁) (f' : α₁ → α₂) (g  …
                    -/
  bimap_bimap := by simp [bimap, functor_norm]
                    /-
                      🎉 no goals
                    -/


instance Sum.bifunctor : Bifunctor Sum where bimap := @Sum.map


instance Sum.lawfulBifunctor : LawfulBifunctor Sum where
                 /-
                   F : Type u₀ → Type u₁ → Type u₂
                   inst✝ : Bifunctor F
                   ⊢ ∀ {α : Type u_1} {β : Type u_2} (x : Sum α β), Eq (Bifunctor.bimap id id x) x
                 -/
  id_bimap := by aesop
                 /-
                   🎉 no goals
                 -/
                    /-
                      F : Type u₀ → Type u₁ → Type u₂
                      inst✝ : Bifunctor F
                      ⊢ ∀ {α₀ α₁ α₂ : Type u_1} {β₀ β₁ β₂ : Type u_2} (f : α₀ → α₁) (f' : α₁ → α₂) ( …
                    -/
  bimap_bimap := by aesop
                    /-
                      🎉 no goals
                    -/


instance (priority := 10) Bifunctor.functor {α} : Functor (F α) where map f x := snd f x


instance (priority := 10) Bifunctor.lawfulFunctor [LawfulBifunctor F] {α} :
    LawfulFunctor (F α) where
  -- Porting note: `mapConst` is required to prove new theorem
               /-
                 F : Type u₀ → Type u₁ → Type u₂
                 inst✝¹ : Bifunctor F
                 inst✝ : LawfulBifunctor F
                 α : Type u₀
                 ⊢ ∀ {α_1 : Type u₁} (x : F α α_1), Eq (Functor.map id x) x
               -/
  id_map := by simp [Functor.map, functor_norm]
                  /-
                    F : Type u₀ → Type u₁ → Type u₂
                    inst✝¹ : Bifunctor F
                    inst✝ : LawfulBifunctor F
                    α : Type u₀
                    ⊢ ∀ {α_1 β : Type u₁}, Eq Functor.mapConst (Function.comp Functor.map (Functio …
                  -/
               /-
                 🎉 no goals
               -/
                  /-
                    🎉 no goals
                  -/
                 /-
                   F : Type u₀ → Type u₁ → Type u₂
                   inst✝¹ : Bifunctor F
                   inst✝ : LawfulBifunctor F
                   α : Type u₀
                   ⊢ ∀ {α_1 β γ : Type u₁} (g : α_1 → β) (h : β → γ) (x : F α α_1), Eq (Functor.m …
                 -/
  comp_map := by simp [Functor.map, functor_norm]
                 /-
                   🎉 no goals
                 -/
  map_const := by simp [mapConst, Functor.map]


instance Function.bicompl.bifunctor : Bifunctor (bicompl F G H) where
  bimap {_α α' _β β'} f f' x := (bimap (map f) (map f') x : F (G α') (H β'))


instance Function.bicompl.lawfulBifunctor [LawfulFunctor G] [LawfulFunctor H] [LawfulBifunctor F] :
    LawfulBifunctor (bicompl F G H) := by
  /-
    F : Type u₀ → Type u₁ → Type u₂
    inst✝⁵ : Bifunctor F
    G : Type u_1 → Type u₀
    H : Type u_2 → Type u₁
    inst✝⁴ : Functor G
    inst✝³ : Functor H
    inst✝² : LawfulFunctor G
    inst✝¹ : LawfulFunctor H
    inst✝ : LawfulBifunctor F
    ⊢ LawfulBifunctor (Function.bicompl F G H)
  -/
                             /-
                               🎉 no goals
                             -/
  constructor <;> intros <;> simp [bimap, map_id, map_comp_map, functor_norm]
                             /-
                               🎉 no goals
                             -/


instance Function.bicompr.bifunctor : Bifunctor (bicompr G F) where
  bimap {_α α' _β β'} f f' x := (map (bimap f f') x : G (F α' β'))


instance Function.bicompr.lawfulBifunctor [LawfulFunctor G] [LawfulBifunctor F] :
    LawfulBifunctor (bicompr G F) := by
  /-
    F : Type u₀ → Type u₁ → Type u₂
    inst✝³ : Bifunctor F
    G : Type u₂ → Type u_1
    inst✝² : Functor G
    inst✝¹ : LawfulFunctor G
    inst✝ : LawfulBifunctor F
    ⊢ LawfulBifunctor (Function.bicompr G F)
  -/
                             /-
                               🎉 no goals
                             -/
  constructor <;> intros <;> simp [bimap, functor_norm]
                             /-
                               🎉 no goals
                             -/


