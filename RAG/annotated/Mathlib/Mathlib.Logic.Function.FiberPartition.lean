/-- The indexing set of the partition. -/
def Fiber (f : Y → Z) : Type _ := Set.range (fun (x : Set.range f) ↦ f ⁻¹' {x.val})


/--
Any `a : Fiber f` is of the form `f ⁻¹' {x}` for some `x` in the image of `f`. We define `a.image`
as an arbitrary such `x`.
-/
noncomputable def image (f : Y → Z) (a : Fiber f) : Z := a.2.choose.1


lemma eq_fiber_image  (f : Y → Z) (a : Fiber f) : a.1 = f ⁻¹' {a.image} := a.2.choose_spec.symm


/--
Given `y : Y`, `Fiber.mk f y` is the fiber of `f` that `y` belongs to, as an element of `Fiber f`.
-/
                                                         /-
                                                           X : Type u_1
                                                           Y : Type u_2
                                                           Z : Type u_3
                                                           f : Y → Z
                                                           y : Y
                                                           ⊢ Membership.mem (Set.range fun x => Set.preimage f (Singleton.singleton ↑x))  …
                                                         -/
def mk (f : Y → Z) (y : Y) : Fiber f := ⟨f ⁻¹' {f y}, by simp⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- `y : Y` as a term of the type `Fiber.mk f y` -/
def mkSelf (f : Y → Z) (y : Y) : (mk f y).val := ⟨y, rfl⟩


lemma map_eq_image (f : Y → Z) (a : Fiber f) (x : a.1) : f x = a.image := by
  /-
    Y : Type u_2
    Z : Type u_3
    f : Y → Z
    a : Function.Fiber f
    x : ↑↑a
    ⊢ Eq (f ↑x) (Function.Fiber.image f a)
  -/
  have := a.2.choose_spec
  /-
    Y : Type u_2
    Z : Type u_3
    f : Y → Z
    a : Function.Fiber f
    x : ↑↑a
    this : Eq ((fun x => Set.preimage f (Singleton.singleton ↑x)) (Exists.choose ⋯ …
    ⊢ Eq (f ↑x) (Function.Fiber.image f a)
  -/
  rw [← Set.mem_singleton_iff, ← Set.mem_preimage]
  /-
    Y : Type u_2
    Z : Type u_3
    f : Y → Z
    a : Function.Fiber f
    x : ↑↑a
    this : Eq ((fun x => Set.preimage f (Singleton.singleton ↑x)) (Exists.choose ⋯ …
    ⊢ Membership.mem (Set.preimage f (Singleton.singleton (Function.Fiber.image f  …
  -/
  convert x.prop
  /-
    🎉 no goals
  -/


lemma mk_image (f : Y → Z) (y : Y) : (Fiber.mk f y).image = f y :=
  (map_eq_image (x := mkSelf f y)).symm


lemma mem_iff_eq_image (f : Y → Z) (y : Y) (a : Fiber f) : y ∈ a.val ↔ f y = a.image :=
                                               /-
                                                 Y : Type u_2
                                                 Z : Type u_3
                                                 f : Y → Z
                                                 y : Y
                                                 a : Function.Fiber f
                                                 h : Eq (f y) (Function.Fiber.image f a)
                                                 ⊢ Membership.mem (↑a) y
                                               -/
  ⟨fun h ↦ a.map_eq_image _ ⟨y, h⟩, fun h ↦ by rw [a.eq_fiber_image]; exact h⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- An arbitrary element of `a : Fiber f`. -/
noncomputable def preimage (f : Y → Z) (a : Fiber f) : Y := a.2.choose.2.choose


lemma map_preimage_eq_image (f : Y → Z) (a : Fiber f) : f a.preimage = a.image :=
  a.2.choose.2.choose_spec


lemma fiber_nonempty (f : Y → Z) (a : Fiber f) : Set.Nonempty a.val := by
  /-
    Y : Type u_2
    Z : Type u_3
    f : Y → Z
    a : Function.Fiber f
    ⊢ (↑a).Nonempty
  -/
  refine ⟨preimage f a, ?_⟩
  /-
    Y : Type u_2
    Z : Type u_3
    f : Y → Z
    a : Function.Fiber f
    ⊢ Membership.mem (↑a) (Function.Fiber.preimage f a)
  -/
  rw [mem_iff_eq_image, ← map_preimage_eq_image]
  /-
    🎉 no goals
  -/


lemma map_preimage_eq_image_map {W : Type*} (f : Y → Z) (g : Z → W) (a : Fiber (g ∘ f)) :
                                     /-
                                       Y : Type u_2
                                       Z : Type u_3
                                       W : Type u_4
                                       f : Y → Z
                                       g : Z → W
                                       a : Function.Fiber (Function.comp g f)
                                       ⊢ Eq (g (f (Function.Fiber.preimage (Function.comp g f) a))) (Function.Fiber.i …
                                     -/
    g (f a.preimage) = a.image := by rw [← map_preimage_eq_image, comp_apply]
                                     /-
                                       🎉 no goals
                                     -/


lemma image_eq_image_mk (f : Y → Z) (g : X → Y) (a : Fiber (f ∘ g)) :
    a.image = (Fiber.mk f (g (a.preimage _))).image := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    f : Y → Z
    g : X → Y
    a : Function.Fiber (Function.comp f g)
    ⊢ Eq (Function.Fiber.image (Function.comp f g) a) (Function.Fiber.image f (Fun …
  -/
  rw [← map_preimage_eq_image_map _ _ a, mk_image]
  /-
    🎉 no goals
  -/


