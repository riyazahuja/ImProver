/-- A dependent function `Π i, β i` with finite support, with notation `Π₀ i, β i`.

Note that `DFinsupp.support` is the preferred API for accessing the support of the function,
`DFinsupp.support'` is an implementation detail that aids computability; see the implementation
notes in this file for more information. -/
structure DFinsupp [∀ i, Zero (β i)] : Type max u v where mk' ::
  /-- The underlying function of a dependent function with finite support (aka `DFinsupp`). -/
  toFun : ∀ i, β i
  /-- The support of a dependent function with finite support (aka `DFinsupp`). -/
  support' : Trunc { s : Multiset ι // ∀ i, i ∈ s ∨ toFun i = 0 }


/-- `Π₀ i, β i` denotes the type of dependent functions with finite support `DFinsupp β`. -/
notation3 "Π₀ "(...)", "r:(scoped f => DFinsupp f) => r


instance instDFunLike : DFunLike (Π₀ i, β i) ι β :=
  ⟨fun f => f.toFun, fun ⟨f₁, s₁⟩ ⟨f₂, s₁⟩ ↦ fun (h : f₁ = f₂) ↦ by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      x✝¹ x✝ : DFinsupp fun i => β i
      f₁ : (i : ι) → β i
      s₁✝ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f₁ i) 0))
      f₂ : (i : ι) → β i
      s₁ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f₂ i) 0))
      h : Eq f₁ f₂
      ⊢ Eq { toFun := f₁, support' := s₁✝ } { toFun := f₂, support' := s₁ }
    -/
    subst h
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      x✝¹ x✝ : DFinsupp fun i => β i
      f₁ : (i : ι) → β i
      s₁✝ s₁ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f₁ i) …
      ⊢ Eq { toFun := f₁, support' := s₁✝ } { toFun := f₁, support' := s₁ }
    -/
    congr
    /-
      case e_support'
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      x✝¹ x✝ : DFinsupp fun i => β i
      f₁ : (i : ι) → β i
      s₁✝ s₁ : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f₁ i) …
      ⊢ Eq s₁✝ s₁
    -/
    subsingleton ⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem toFun_eq_coe (f : Π₀ i, β i) : f.toFun = f :=
  rfl


@[ext]
theorem ext {f g : Π₀ i, β i} (h : ∀ i, f i = g i) : f = g :=
  DFunLike.ext _ _ h


lemma ne_iff {f g : Π₀ i, β i} : f ≠ g ↔ ∃ i, f i ≠ g i := DFunLike.ne_iff


instance : Zero (Π₀ i, β i) :=
  ⟨⟨0, Trunc.mk <| ⟨∅, fun _ => Or.inr rfl⟩⟩⟩


instance : Inhabited (Π₀ i, β i) :=
  ⟨0⟩


@[simp, norm_cast] lemma coe_mk' (f : ∀ i, β i) (s) : ⇑(⟨f, s⟩ : Π₀ i, β i) = f := rfl


@[simp, norm_cast] lemma coe_zero : ⇑(0 : Π₀ i, β i) = 0 := rfl


theorem zero_apply (i : ι) : (0 : Π₀ i, β i) i = 0 :=
  rfl


/-- The composition of `f : β₁ → β₂` and `g : Π₀ i, β₁ i` is
  `mapRange f hf g : Π₀ i, β₂ i`, well defined when `f 0 = 0`.

This preserves the structure on `f`, and exists in various bundled forms for when `f` is itself
bundled:

* `DFinsupp.mapRange.addMonoidHom`
* `DFinsupp.mapRange.addEquiv`
* `dfinsupp.mapRange.linearMap`
* `dfinsupp.mapRange.linearEquiv`
-/
def mapRange (f : ∀ i, β₁ i → β₂ i) (hf : ∀ i, f i 0 = 0) (x : Π₀ i, β₁ i) : Π₀ i, β₂ i :=
  ⟨fun i => f i (x i),
    x.support'.map fun s => ⟨s.1, fun i => (s.2 i).imp_right fun h : x i = 0 => by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i
        hf : ∀ (i : ι), Eq (f i 0) 0
        x : DFinsupp fun i => β₁ i
        s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        i : ι
        h : Eq (x i) 0
        ⊢ Eq ((fun i => f i (x i)) i) 0
      -/
      rw [← hf i, ← h]⟩⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem mapRange_apply (f : ∀ i, β₁ i → β₂ i) (hf : ∀ i, f i 0 = 0) (g : Π₀ i, β₁ i) (i : ι) :
    mapRange f hf g i = f i (g i) :=
  rfl


@[simp]
theorem mapRange_id (h : ∀ i, id (0 : β₁ i) = 0 := fun _ => rfl) (g : Π₀ i : ι, β₁ i) :
    mapRange (fun i => (id : β₁ i → β₁ i)) h g = g := by
  /-
    ι : Type u
    β₁ : ι → Type v₁
    inst✝ : (i : ι) → Zero (β₁ i)
    h : optParam (∀ (i : ι), Eq (id 0) 0) ⋯
    g : DFinsupp fun i => β₁ i
    ⊢ Eq (DFinsupp.mapRange (fun i => id) h g) g
  -/
  ext
  /-
    case h
    ι : Type u
    β₁ : ι → Type v₁
    inst✝ : (i : ι) → Zero (β₁ i)
    h : optParam (∀ (i : ι), Eq (id 0) 0) ⋯
    g : DFinsupp fun i => β₁ i
    i✝ : ι
    ⊢ Eq ((DFinsupp.mapRange (fun i => id) h g) i✝) (g i✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mapRange_comp (f : ∀ i, β₁ i → β₂ i) (f₂ : ∀ i, β i → β₁ i) (hf : ∀ i, f i 0 = 0)
    (hf₂ : ∀ i, f₂ i 0 = 0) (h : ∀ i, (f i ∘ f₂ i) 0 = 0) (g : Π₀ i : ι, β i) :
    mapRange (fun i => f i ∘ f₂ i) h g = mapRange f hf (mapRange f₂ hf₂ g) := by
  /-
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    f₂ : (i : ι) → β i → β₁ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    hf₂ : ∀ (i : ι), Eq (f₂ i 0) 0
    h : ∀ (i : ι), Eq (Function.comp (f i) (f₂ i) 0) 0
    g : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.mapRange (fun i => Function.comp (f i) (f₂ i)) h g) (DFinsupp.m …
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    f₂ : (i : ι) → β i → β₁ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    hf₂ : ∀ (i : ι), Eq (f₂ i 0) 0
    h : ∀ (i : ι), Eq (Function.comp (f i) (f₂ i) 0) 0
    g : DFinsupp fun i => β i
    i✝ : ι
    ⊢ Eq ((DFinsupp.mapRange (fun i => Function.comp (f i) (f₂ i)) h g) i✝) ((DFin …
  -/
  simp only [mapRange_apply]; rfl
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem mapRange_zero (f : ∀ i, β₁ i → β₂ i) (hf : ∀ i, f i 0 = 0) :
    mapRange f hf (0 : Π₀ i, β₁ i) = 0 := by
  /-
    ι : Type u
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    ⊢ Eq (DFinsupp.mapRange f hf 0) 0
  -/
  ext
  /-
    case h
    ι : Type u
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    i✝ : ι
    ⊢ Eq ((DFinsupp.mapRange f hf 0) i✝) (0 i✝)
  -/
  simp only [mapRange_apply, coe_zero, Pi.zero_apply, hf]
  /-
    🎉 no goals
  -/


/-- Let `f i` be a binary operation `β₁ i → β₂ i → β i` such that `f i 0 0 = 0`.
Then `zipWith f hf` is a binary operation `Π₀ i, β₁ i → Π₀ i, β₂ i → Π₀ i, β i`. -/
def zipWith (f : ∀ i, β₁ i → β₂ i → β i) (hf : ∀ i, f i 0 0 = 0) (x : Π₀ i, β₁ i) (y : Π₀ i, β₂ i) :
    Π₀ i, β i :=
  ⟨fun i => f i (x i) (y i), by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      x : DFinsupp fun i => β₁ i
      y : DFinsupp fun i => β₂ i
      ⊢ Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq ((fun i => f  …
    -/
    refine x.support'.bind fun xs => ?_
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      x : DFinsupp fun i => β₁ i
      y : DFinsupp fun i => β₂ i
      xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
      ⊢ Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq ((fun i => f  …
    -/
    refine y.support'.map fun ys => ?_
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      x : DFinsupp fun i => β₁ i
      y : DFinsupp fun i => β₂ i
      xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
      ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
      ⊢ Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq ((fun i => f i (x i) …
    -/
    refine ⟨xs + ys, fun i => ?_⟩
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      x : DFinsupp fun i => β₁ i
      y : DFinsupp fun i => β₂ i
      xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
      ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
      i : ι
      ⊢ Or (Membership.mem (HAdd.hAdd ↑xs ↑ys) i) (Eq ((fun i => f i (x i) (y i)) i) …
    -/
    obtain h1 | (h1 : x i = 0) := xs.prop i
      /-
        case inl
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Membership.mem (↑xs) i
        ⊢ Or (Membership.mem (HAdd.hAdd ↑xs ↑ys) i) (Eq ((fun i => f i (x i) (y i)) i) …
      -/
    · left
      /-
        case inl.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Membership.mem (↑xs) i
        ⊢ Membership.mem (HAdd.hAdd ↑xs ↑ys) i
      -/
      rw [Multiset.mem_add]
      /-
        case inl.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Membership.mem (↑xs) i
        ⊢ Or (Membership.mem (↑xs) i) (Membership.mem (↑ys) i)
      -/
      left
      /-
        case inl.h.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Membership.mem (↑xs) i
        ⊢ Membership.mem (↑xs) i
      -/
      exact h1
      /-
        🎉 no goals
      -/
    /-
      case inr
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      x : DFinsupp fun i => β₁ i
      y : DFinsupp fun i => β₂ i
      xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
      ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
      i : ι
      h1 : Eq (x i) 0
      ⊢ Or (Membership.mem (HAdd.hAdd ↑xs ↑ys) i) (Eq ((fun i => f i (x i) (y i)) i) …
    -/
    obtain h2 | (h2 : y i = 0) := ys.prop i
      /-
        case inr.inl
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Eq (x i) 0
        h2 : Membership.mem (↑ys) i
        ⊢ Or (Membership.mem (HAdd.hAdd ↑xs ↑ys) i) (Eq ((fun i => f i (x i) (y i)) i) …
      -/
    · left
      /-
        case inr.inl.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Eq (x i) 0
        h2 : Membership.mem (↑ys) i
        ⊢ Membership.mem (HAdd.hAdd ↑xs ↑ys) i
      -/
      rw [Multiset.mem_add]
      /-
        case inr.inl.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Eq (x i) 0
        h2 : Membership.mem (↑ys) i
        ⊢ Or (Membership.mem (↑xs) i) (Membership.mem (↑ys) i)
      -/
      right
      /-
        case inr.inl.h.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i → β i
        hf : ∀ (i : ι), Eq (f i 0 0) 0
        x : DFinsupp fun i => β₁ i
        y : DFinsupp fun i => β₂ i
        xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
        ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
        i : ι
        h1 : Eq (x i) 0
        h2 : Membership.mem (↑ys) i
        ⊢ Membership.mem (↑ys) i
      -/
      exact h2
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      x : DFinsupp fun i => β₁ i
      y : DFinsupp fun i => β₂ i
      xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
      ys : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (y.toFun i) 0)
      i : ι
      h1 : Eq (x i) 0
      h2 : Eq (y i) 0
      ⊢ Or (Membership.mem (HAdd.hAdd ↑xs ↑ys) i) (Eq ((fun i => f i (x i) (y i)) i) …
    -/
    right; rw [← hf, ← h1, ← h2]⟩
           /-
             🎉 no goals
           -/


@[simp]
theorem zipWith_apply (f : ∀ i, β₁ i → β₂ i → β i) (hf : ∀ i, f i 0 0 = 0) (g₁ : Π₀ i, β₁ i)
    (g₂ : Π₀ i, β₂ i) (i : ι) : zipWith f hf g₁ g₂ i = f i (g₁ i) (g₂ i) :=
  rfl


/-- `x.piecewise y s` is the finitely supported function equal to `x` on the set `s`,
  and to `y` on its complement. -/
def piecewise : Π₀ i, β i :=
  zipWith (fun i x y => if i ∈ s then x else y) (fun _ => ite_self 0) x y


theorem piecewise_apply (i : ι) : x.piecewise y s i = if i ∈ s then x i else y i :=
  zipWith_apply _ _ x y i


@[simp, norm_cast]
theorem coe_piecewise : ⇑(x.piecewise y s) = s.piecewise x y := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    x y : DFinsupp fun i => β i
    s : Set ι
    inst✝ : (i : ι) → Decidable (Membership.mem s i)
    ⊢ Eq (⇑(x.piecewise y s)) (s.piecewise ⇑x ⇑y)
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    x y : DFinsupp fun i => β i
    s : Set ι
    inst✝ : (i : ι) → Decidable (Membership.mem s i)
    x✝ : ι
    ⊢ Eq ((x.piecewise y s) x✝) (s.piecewise (⇑x) (⇑y) x✝)
  -/
  apply piecewise_apply
  /-
    🎉 no goals
  -/


instance [∀ i, AddZeroClass (β i)] : Add (Π₀ i, β i) :=
  ⟨zipWith (fun _ => (· + ·)) fun _ => add_zero 0⟩


theorem add_apply [∀ i, AddZeroClass (β i)] (g₁ g₂ : Π₀ i, β i) (i : ι) :
    (g₁ + g₂) i = g₁ i + g₂ i :=
  rfl


@[simp, norm_cast]
theorem coe_add [∀ i, AddZeroClass (β i)] (g₁ g₂ : Π₀ i, β i) : ⇑(g₁ + g₂) = g₁ + g₂ :=
  rfl


instance addZeroClass [∀ i, AddZeroClass (β i)] : AddZeroClass (Π₀ i, β i) :=
  DFunLike.coe_injective.addZeroClass _ coe_zero coe_add


instance instIsLeftCancelAdd [∀ i, AddZeroClass (β i)] [∀ i, IsLeftCancelAdd (β i)] :
    IsLeftCancelAdd (Π₀ i, β i) where
  add_left_cancel _ _ _ h := ext fun x => add_left_cancel <| DFunLike.congr_fun h x


instance instIsRightCancelAdd [∀ i, AddZeroClass (β i)] [∀ i, IsRightCancelAdd (β i)] :
    IsRightCancelAdd (Π₀ i, β i) where
  add_right_cancel _ _ _ h := ext fun x => add_right_cancel <| DFunLike.congr_fun h x


instance instIsCancelAdd [∀ i, AddZeroClass (β i)] [∀ i, IsCancelAdd (β i)] :
    IsCancelAdd (Π₀ i, β i) where


/-- Note the general `SMul` instance doesn't apply as `ℕ` is not distributive
unless `β i`'s addition is commutative. -/
instance hasNatScalar [∀ i, AddMonoid (β i)] : SMul ℕ (Π₀ i, β i) :=
  ⟨fun c v => v.mapRange (fun _ => (c • ·)) fun _ => nsmul_zero _⟩


theorem nsmul_apply [∀ i, AddMonoid (β i)] (b : ℕ) (v : Π₀ i, β i) (i : ι) : (b • v) i = b • v i :=
  rfl


@[simp, norm_cast]
theorem coe_nsmul [∀ i, AddMonoid (β i)] (b : ℕ) (v : Π₀ i, β i) : ⇑(b • v) = b • ⇑v :=
  rfl


instance [∀ i, AddMonoid (β i)] : AddMonoid (Π₀ i, β i) :=
  DFunLike.coe_injective.addMonoid _ coe_zero coe_add fun _ _ => coe_nsmul _ _


/-- Coercion from a `DFinsupp` to a pi type is an `AddMonoidHom`. -/
def coeFnAddMonoidHom [∀ i, AddZeroClass (β i)] : (Π₀ i, β i) →+ ∀ i, β i where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


instance addCommMonoid [∀ i, AddCommMonoid (β i)] : AddCommMonoid (Π₀ i, β i) :=
  DFunLike.coe_injective.addCommMonoid _ coe_zero coe_add fun _ _ => coe_nsmul _ _


instance [∀ i, AddGroup (β i)] : Neg (Π₀ i, β i) :=
  ⟨fun f => f.mapRange (fun _ => Neg.neg) fun _ => neg_zero⟩


theorem neg_apply [∀ i, AddGroup (β i)] (g : Π₀ i, β i) (i : ι) : (-g) i = -g i :=
  rfl


@[simp, norm_cast] lemma coe_neg [∀ i, AddGroup (β i)] (g : Π₀ i, β i) : ⇑(-g) = -g := rfl


instance [∀ i, AddGroup (β i)] : Sub (Π₀ i, β i) :=
  ⟨zipWith (fun _ => Sub.sub) fun _ => sub_zero 0⟩


theorem sub_apply [∀ i, AddGroup (β i)] (g₁ g₂ : Π₀ i, β i) (i : ι) : (g₁ - g₂) i = g₁ i - g₂ i :=
  rfl


@[simp, norm_cast]
theorem coe_sub [∀ i, AddGroup (β i)] (g₁ g₂ : Π₀ i, β i) : ⇑(g₁ - g₂) = g₁ - g₂ :=
  rfl


/-- Note the general `SMul` instance doesn't apply as `ℤ` is not distributive
unless `β i`'s addition is commutative. -/
instance hasIntScalar [∀ i, AddGroup (β i)] : SMul ℤ (Π₀ i, β i) :=
  ⟨fun c v => v.mapRange (fun _ => (c • ·)) fun _ => zsmul_zero _⟩


theorem zsmul_apply [∀ i, AddGroup (β i)] (b : ℤ) (v : Π₀ i, β i) (i : ι) : (b • v) i = b • v i :=
  rfl


@[simp, norm_cast]
theorem coe_zsmul [∀ i, AddGroup (β i)] (b : ℤ) (v : Π₀ i, β i) : ⇑(b • v) = b • ⇑v :=
  rfl


instance [∀ i, AddGroup (β i)] : AddGroup (Π₀ i, β i) :=
  DFunLike.coe_injective.addGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => coe_nsmul _ _)
    fun _ _ => coe_zsmul _ _


instance addCommGroup [∀ i, AddCommGroup (β i)] : AddCommGroup (Π₀ i, β i) :=
  DFunLike.coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => coe_nsmul _ _)
    fun _ _ => coe_zsmul _ _


/-- `Filter p f` is the function which is `f i` if `p i` is true and 0 otherwise. -/
def filter [∀ i, Zero (β i)] (p : ι → Prop) [DecidablePred p] (x : Π₀ i, β i) : Π₀ i, β i :=
  ⟨fun i => if p i then x i else 0,
    x.support'.map fun xs =>
                                                                  /-
                                                                    ι : Type u
                                                                    γ : Type w
                                                                    β : ι → Type v
                                                                    β₁ : ι → Type v₁
                                                                    β₂ : ι → Type v₂
                                                                    inst✝¹ : (i : ι) → Zero (β i)
                                                                    p : ι → Prop
                                                                    inst✝ : DecidablePred p
                                                                    x : DFinsupp fun i => β i
                                                                    xs : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (x.toFun i) 0)
                                                                    i : ι
                                                                    H : Eq (x i) 0
                                                                    ⊢ Eq ((fun i => ite (p i) (x i) 0) i) 0
                                                                  -/
      ⟨xs.1, fun i => (xs.prop i).imp_right fun H : x i = 0 => by simp only [H, ite_self]⟩⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem filter_apply [∀ i, Zero (β i)] (p : ι → Prop) [DecidablePred p] (i : ι) (f : Π₀ i, β i) :
    f.filter p i = if p i then f i else 0 :=
  rfl


theorem filter_apply_pos [∀ i, Zero (β i)] {p : ι → Prop} [DecidablePred p] (f : Π₀ i, β i) {i : ι}
                                         /-
                                           ι : Type u
                                           β : ι → Type v
                                           inst✝¹ : (i : ι) → Zero (β i)
                                           p : ι → Prop
                                           inst✝ : DecidablePred p
                                           f : DFinsupp fun i => β i
                                           i : ι
                                           h : p i
                                           ⊢ Eq ((DFinsupp.filter p f) i) (f i)
                                         -/
    (h : p i) : f.filter p i = f i := by simp only [filter_apply, if_pos h]
                                         /-
                                           🎉 no goals
                                         -/


theorem filter_apply_neg [∀ i, Zero (β i)] {p : ι → Prop} [DecidablePred p] (f : Π₀ i, β i) {i : ι}
                                        /-
                                          ι : Type u
                                          β : ι → Type v
                                          inst✝¹ : (i : ι) → Zero (β i)
                                          p : ι → Prop
                                          inst✝ : DecidablePred p
                                          f : DFinsupp fun i => β i
                                          i : ι
                                          h : Not (p i)
                                          ⊢ Eq ((DFinsupp.filter p f) i) 0
                                        -/
    (h : ¬p i) : f.filter p i = 0 := by simp only [filter_apply, if_neg h]
                                        /-
                                          🎉 no goals
                                        -/


theorem filter_pos_add_filter_neg [∀ i, AddZeroClass (β i)] (f : Π₀ i, β i) (p : ι → Prop)
    [DecidablePred p] : (f.filter p + f.filter fun i => ¬p i) = f :=
  ext fun i => by
    /-
      ι : Type u
      β : ι → Type v
      inst✝¹ : (i : ι) → AddZeroClass (β i)
      f : DFinsupp fun i => β i
      p : ι → Prop
      inst✝ : DecidablePred p
      i : ι
      ⊢ Eq ((HAdd.hAdd (DFinsupp.filter p f) (DFinsupp.filter (fun i => Not (p i)) f …
    -/
                                                       /-
                                                         🎉 no goals
                                                       -/
    simp only [add_apply, filter_apply]; split_ifs <;> simp only [add_zero, zero_add]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem filter_zero [∀ i, Zero (β i)] (p : ι → Prop) [DecidablePred p] :
    (0 : Π₀ i, β i).filter p = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    p : ι → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (DFinsupp.filter p 0) 0
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    p : ι → Prop
    inst✝ : DecidablePred p
    i✝ : ι
    ⊢ Eq ((DFinsupp.filter p 0) i✝) (0 i✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem filter_add [∀ i, AddZeroClass (β i)] (p : ι → Prop) [DecidablePred p] (f g : Π₀ i, β i) :
    (f + g).filter p = f.filter p + g.filter p := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    p : ι → Prop
    inst✝ : DecidablePred p
    f g : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.filter p (HAdd.hAdd f g)) (HAdd.hAdd (DFinsupp.filter p f) (DFi …
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → AddZeroClass (β i)
    p : ι → Prop
    inst✝ : DecidablePred p
    f g : DFinsupp fun i => β i
    i✝ : ι
    ⊢ Eq ((DFinsupp.filter p (HAdd.hAdd f g)) i✝) ((HAdd.hAdd (DFinsupp.filter p f …
  -/
  simp [ite_add_zero]
  /-
    🎉 no goals
  -/


/-- `DFinsupp.filter` as an `AddMonoidHom`. -/
@[simps]
def filterAddMonoidHom [∀ i, AddZeroClass (β i)] (p : ι → Prop) [DecidablePred p] :
    (Π₀ i, β i) →+ Π₀ i, β i where
  toFun := filter p
  map_zero' := filter_zero p
  map_add' := filter_add p


@[simp]
theorem filter_neg [∀ i, AddGroup (β i)] (p : ι → Prop) [DecidablePred p] (f : Π₀ i, β i) :
    (-f).filter p = -f.filter p :=
  (filterAddMonoidHom β p).map_neg f


@[simp]
theorem filter_sub [∀ i, AddGroup (β i)] (p : ι → Prop) [DecidablePred p] (f g : Π₀ i, β i) :
    (f - g).filter p = f.filter p - g.filter p :=
  (filterAddMonoidHom β p).map_sub f g


/-- `subtypeDomain p f` is the restriction of the finitely supported function
  `f` to the subtype `p`. -/
def subtypeDomain [∀ i, Zero (β i)] (p : ι → Prop) [DecidablePred p] (x : Π₀ i, β i) :
    Π₀ i : Subtype p, β i :=
  ⟨fun i => x (i : ι),
    x.support'.map fun xs =>
      ⟨(Multiset.filter p xs.1).attach.map fun j => ⟨j.1, (Multiset.mem_filter.1 j.2).2⟩, fun i =>
        (xs.prop i).imp_left fun H =>
          Multiset.mem_map.2
            ⟨⟨i, Multiset.mem_filter.2 ⟨H, i.2⟩⟩, Multiset.mem_attach _ _, Subtype.eta _ _⟩⟩⟩


@[simp]
theorem subtypeDomain_zero [∀ i, Zero (β i)] {p : ι → Prop} [DecidablePred p] :
    subtypeDomain p (0 : Π₀ i, β i) = 0 :=
  rfl


@[simp]
theorem subtypeDomain_apply [∀ i, Zero (β i)] {p : ι → Prop} [DecidablePred p] {i : Subtype p}
    {v : Π₀ i, β i} : (subtypeDomain p v) i = v i :=
  rfl


@[simp]
theorem subtypeDomain_add [∀ i, AddZeroClass (β i)] {p : ι → Prop} [DecidablePred p]
    (v v' : Π₀ i, β i) : (v + v').subtypeDomain p = v.subtypeDomain p + v'.subtypeDomain p :=
  DFunLike.coe_injective rfl


/-- `subtypeDomain` but as an `AddMonoidHom`. -/
@[simps]
def subtypeDomainAddMonoidHom [∀ i, AddZeroClass (β i)] (p : ι → Prop) [DecidablePred p] :
    (Π₀ i : ι, β i) →+ Π₀ i : Subtype p, β i where
  toFun := subtypeDomain p
  map_zero' := subtypeDomain_zero
  map_add' := subtypeDomain_add


@[simp]
theorem subtypeDomain_neg [∀ i, AddGroup (β i)] {p : ι → Prop} [DecidablePred p] {v : Π₀ i, β i} :
    (-v).subtypeDomain p = -v.subtypeDomain p :=
  DFunLike.coe_injective rfl


@[simp]
theorem subtypeDomain_sub [∀ i, AddGroup (β i)] {p : ι → Prop} [DecidablePred p]
    {v v' : Π₀ i, β i} : (v - v').subtypeDomain p = v.subtypeDomain p - v'.subtypeDomain p :=
  DFunLike.coe_injective rfl


theorem finite_support (f : Π₀ i, β i) : Set.Finite { i | f i ≠ 0 } :=
  Trunc.induction_on f.support' fun xs ↦
    xs.1.finite_toSet.subset fun i H ↦ ((xs.prop i).resolve_right H)


/-- Create an element of `Π₀ i, β i` from a finset `s` and a function `x`
defined on this `Finset`. -/
def mk (s : Finset ι) (x : ∀ i : (↑s : Set ι), β (i : ι)) : Π₀ i, β i :=
  ⟨fun i => if H : i ∈ s then x ⟨i, H⟩ else 0,
    Trunc.mk ⟨s.1, fun i => if H : i ∈ s then Or.inl H else Or.inr <| dif_neg H⟩⟩


@[simp]
theorem mk_apply : (mk s x : ∀ i, β i) i = if H : i ∈ s then x ⟨i, H⟩ else 0 :=
  rfl


theorem mk_of_mem (hi : i ∈ s) : (mk s x : ∀ i, β i) i = x ⟨i, hi⟩ :=
  dif_pos hi


theorem mk_of_not_mem (hi : i ∉ s) : (mk s x : ∀ i, β i) i = 0 :=
  dif_neg hi


theorem mk_injective (s : Finset ι) : Function.Injective (@mk ι β _ _ s) := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    ⊢ Function.Injective (DFinsupp.mk s)
  -/
  intro x y H
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    x y : (i : ↑↑s) → β ↑i
    H : Eq (DFinsupp.mk s x) (DFinsupp.mk s y)
    ⊢ Eq x y
  -/
  ext i
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    x y : (i : ↑↑s) → β ↑i
    H : Eq (DFinsupp.mk s x) (DFinsupp.mk s y)
    i : ↑↑s
    ⊢ Eq (x i) (y i)
  -/
  have h1 : (mk s x : ∀ i, β i) i = (mk s y : ∀ i, β i) i := by rw [H]
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    x y : (i : ↑↑s) → β ↑i
    H : Eq (DFinsupp.mk s x) (DFinsupp.mk s y)
    i : ↑↑s
    h1 : Eq ((DFinsupp.mk s x) ↑i) ((DFinsupp.mk s y) ↑i)
    ⊢ Eq (x i) (y i)
  -/
  obtain ⟨i, hi : i ∈ s⟩ := i
  /-
    case h.mk
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    x y : (i : ↑↑s) → β ↑i
    H : Eq (DFinsupp.mk s x) (DFinsupp.mk s y)
    i : ι
    hi : Membership.mem s i
    h1 : Eq ((DFinsupp.mk s x) ↑⟨i, hi⟩) ((DFinsupp.mk s y) ↑⟨i, hi⟩)
    ⊢ Eq (x ⟨i, hi⟩) (y ⟨i, hi⟩)
  -/
  dsimp only [mk_apply, Subtype.coe_mk] at h1
  /-
    case h.mk
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    s : Finset ι
    x y : (i : ↑↑s) → β ↑i
    H : Eq (DFinsupp.mk s x) (DFinsupp.mk s y)
    i : ι
    hi : Membership.mem s i
    h1 : Eq (dite (Membership.mem s i) (fun H => x ⟨i, H⟩) fun H => 0) (dite (Memb …
    ⊢ Eq (x ⟨i, hi⟩) (y ⟨i, hi⟩)
  -/
  simpa only [dif_pos hi] using h1
  /-
    🎉 no goals
  -/


instance unique [∀ i, Subsingleton (β i)] : Unique (Π₀ i, β i) :=
  DFunLike.coe_injective.unique


instance uniqueOfIsEmpty [IsEmpty ι] : Unique (Π₀ i, β i) :=
  DFunLike.coe_injective.unique


/-- Given `Fintype ι`, `equivFunOnFintype` is the `Equiv` between `Π₀ i, β i` and `Π i, β i`.
  (All dependent functions on a finite type are finitely supported.) -/
@[simps apply]
def equivFunOnFintype [Fintype ι] : (Π₀ i, β i) ≃ ∀ i, β i where
  toFun := (⇑)
  invFun f := ⟨f, Trunc.mk ⟨Finset.univ.1, fun _ => Or.inl <| Finset.mem_univ_val _⟩⟩
  left_inv _ := DFunLike.coe_injective rfl
  right_inv _ := rfl


@[simp]
theorem equivFunOnFintype_symm_coe [Fintype ι] (f : Π₀ i, β i) : equivFunOnFintype.symm f = f :=
  Equiv.symm_apply_apply _ _


/-- The function `single i b : Π₀ i, β i` sends `i` to `b`
and all other points to `0`. -/
def single (i : ι) (b : β i) : Π₀ i, β i :=
  ⟨Pi.single i b,
                                                             /-
                                                               ι : Type u
                                                               γ : Type w
                                                               β : ι → Type v
                                                               β₁ : ι → Type v₁
                                                               β₂ : ι → Type v₂
                                                               inst✝¹ : (i : ι) → Zero (β i)
                                                               inst✝ : DecidableEq ι
                                                               i : ι
                                                               b : β i
                                                               j : ι
                                                               ⊢ Eq j i → Membership.mem (Singleton.singleton i) j
                                                             -/
    Trunc.mk ⟨{i}, fun j => (Decidable.eq_or_ne j i).imp (by simp) fun h => Pi.single_eq_of_ne h _⟩⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem single_eq_pi_single {i b} : ⇑(single i b : Π₀ i, β i) = Pi.single i b :=
  rfl


@[simp]
theorem single_apply {i i' b} :
    (single i b : Π₀ i, β i) i' = if h : i = i' then Eq.recOn h b else 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i i' : ι
    b : β i
    ⊢ Eq ((DFinsupp.single i b) i') (dite (Eq i i') (fun h => Eq.recOn h b) fun h  …
  -/
  rw [single_eq_pi_single, Pi.single, Function.update]
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i i' : ι
    b : β i
    ⊢ Eq (dite (Eq i' i) (fun h => Eq.ndrec b ⋯) fun h => 0 i') (dite (Eq i i') (f …
  -/
  simp [@eq_comm _ i i']
  /-
    🎉 no goals
  -/


@[simp]
theorem single_zero (i) : (single i 0 : Π₀ i, β i) = 0 :=
  DFunLike.coe_injective <| Pi.single_zero _


theorem single_eq_same {i b} : (single i b : Π₀ i, β i) i = b := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i : ι
    b : β i
    ⊢ Eq ((DFinsupp.single i b) i) b
  -/
  simp only [single_apply, dite_eq_ite, ite_true]
  /-
    🎉 no goals
  -/


theorem single_eq_of_ne {i i' b} (h : i ≠ i') : (single i b : Π₀ i, β i) i' = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i i' : ι
    b : β i
    h : Ne i i'
    ⊢ Eq ((DFinsupp.single i b) i') 0
  -/
  simp only [single_apply, dif_neg h]
  /-
    🎉 no goals
  -/


theorem single_injective {i} : Function.Injective (single i : β i → Π₀ i, β i) := fun _ _ H =>
  Pi.single_injective β i <| DFunLike.coe_injective.eq_iff.mpr H


/-- Like `Finsupp.single_eq_single_iff`, but with a `HEq` due to dependent types -/
theorem single_eq_single_iff (i j : ι) (xi : β i) (xj : β j) :
    DFinsupp.single i xi = DFinsupp.single j xj ↔ i = j ∧ HEq xi xj ∨ xi = 0 ∧ xj = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i j : ι
    xi : β i
    xj : β j
    ⊢ Iff (Eq (DFinsupp.single i xi) (DFinsupp.single j xj)) (Or (And (Eq i j) (HE …
  -/
  constructor
    /-
      case mp
      ι : Type u
      β : ι → Type v
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : DecidableEq ι
      i j : ι
      xi : β i
      xj : β j
      ⊢ Eq (DFinsupp.single i xi) (DFinsupp.single j xj) → Or (And (Eq i j) (HEq xi  …
    -/
  · intro h
    /-
      case mp
      ι : Type u
      β : ι → Type v
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : DecidableEq ι
      i j : ι
      xi : β i
      xj : β j
      h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
      ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
    -/
    by_cases hij : i = j
      /-
        case pos
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Eq i j
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
    · subst hij
      /-
        case pos
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i : ι
        xi xj : β i
        h : Eq (DFinsupp.single i xi) (DFinsupp.single i xj)
        ⊢ Or (And (Eq i i) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      exact Or.inl ⟨rfl, heq_of_eq (DFinsupp.single_injective h)⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
    · have h_coe : ⇑(DFinsupp.single i xi) = DFinsupp.single j xj := congr_arg (⇑) h
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        h_coe : Eq ⇑(DFinsupp.single i xi) ⇑(DFinsupp.single j xj)
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      have hci := congr_fun h_coe i
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        h_coe : Eq ⇑(DFinsupp.single i xi) ⇑(DFinsupp.single j xj)
        hci : Eq ((DFinsupp.single i xi) i) ((DFinsupp.single j xj) i)
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      have hcj := congr_fun h_coe j
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        h_coe : Eq ⇑(DFinsupp.single i xi) ⇑(DFinsupp.single j xj)
        hci : Eq ((DFinsupp.single i xi) i) ((DFinsupp.single j xj) i)
        hcj : Eq ((DFinsupp.single i xi) j) ((DFinsupp.single j xj) j)
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      rw [DFinsupp.single_eq_same] at hci hcj
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        h_coe : Eq ⇑(DFinsupp.single i xi) ⇑(DFinsupp.single j xj)
        hci : Eq xi ((DFinsupp.single j xj) i)
        hcj : Eq ((DFinsupp.single i xi) j) xj
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      rw [DFinsupp.single_eq_of_ne (Ne.symm hij)] at hci
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        h_coe : Eq ⇑(DFinsupp.single i xi) ⇑(DFinsupp.single j xj)
        hci : Eq xi 0
        hcj : Eq ((DFinsupp.single i xi) j) xj
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      rw [DFinsupp.single_eq_of_ne hij] at hcj
      /-
        case neg
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        h : Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
        hij : Not (Eq i j)
        h_coe : Eq ⇑(DFinsupp.single i xi) ⇑(DFinsupp.single j xj)
        hci : Eq xi 0
        hcj : Eq 0 xj
        ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0))
      -/
      exact Or.inr ⟨hci, hcj.symm⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u
      β : ι → Type v
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : DecidableEq ι
      i j : ι
      xi : β i
      xj : β j
      ⊢ Or (And (Eq i j) (HEq xi xj)) (And (Eq xi 0) (Eq xj 0)) → Eq (DFinsupp.singl …
    -/
  · rintro (⟨rfl, hxi⟩ | ⟨hi, hj⟩)
      /-
        case mpr.inl.intro
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i : ι
        xi xj : β i
        hxi : HEq xi xj
        ⊢ Eq (DFinsupp.single i xi) (DFinsupp.single i xj)
      -/
    · rw [eq_of_heq hxi]
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        ι : Type u
        β : ι → Type v
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : DecidableEq ι
        i j : ι
        xi : β i
        xj : β j
        hi : Eq xi 0
        hj : Eq xj 0
        ⊢ Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
      -/
    · rw [hi, hj, DFinsupp.single_zero, DFinsupp.single_zero]
      /-
        🎉 no goals
      -/


/-- `DFinsupp.single a b` is injective in `a`. For the statement that it is injective in `b`, see
`DFinsupp.single_injective` -/
theorem single_left_injective {b : ∀ i : ι, β i} (h : ∀ i, b i ≠ 0) :
    Function.Injective (fun i => single i (b i) : ι → Π₀ i, β i) := fun _ _ H =>
  (((single_eq_single_iff _ _ _ _).mp H).resolve_right fun hb => h _ hb.1).left


@[simp]
theorem single_eq_zero {i : ι} {xi : β i} : single i xi = 0 ↔ xi = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i : ι
    xi : β i
    ⊢ Iff (Eq (DFinsupp.single i xi) 0) (Eq xi 0)
  -/
  rw [← single_zero i, single_eq_single_iff]
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i : ι
    xi : β i
    ⊢ Iff (Or (And (Eq i i) (HEq xi 0)) (And (Eq xi 0) (Eq 0 0))) (Eq xi 0)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem filter_single (p : ι → Prop) [DecidablePred p] (i : ι) (x : β i) :
    (single i x).filter p = if p i then single i x else 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    p : ι → Prop
    inst✝ : DecidablePred p
    i : ι
    x : β i
    ⊢ Eq (DFinsupp.filter p (DFinsupp.single i x)) (ite (p i) (DFinsupp.single i x …
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    p : ι → Prop
    inst✝ : DecidablePred p
    i : ι
    x : β i
    j : ι
    ⊢ Eq ((DFinsupp.filter p (DFinsupp.single i x)) j) ((ite (p i) (DFinsupp.singl …
  -/
  have := apply_ite (fun x : Π₀ i, β i => x j) (p i) (single i x) 0
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    p : ι → Prop
    inst✝ : DecidablePred p
    i : ι
    x : β i
    j : ι
    this : Eq ((ite (p i) (DFinsupp.single i x) 0) j) (ite (p i) ((DFinsupp.single …
    ⊢ Eq ((DFinsupp.filter p (DFinsupp.single i x)) j) ((ite (p i) (DFinsupp.singl …
  -/
  dsimp at this
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    p : ι → Prop
    inst✝ : DecidablePred p
    i : ι
    x : β i
    j : ι
    this : Eq ((ite (p i) (DFinsupp.single i x) 0) j) (ite (p i) ((DFinsupp.single …
    ⊢ Eq ((DFinsupp.filter p (DFinsupp.single i x)) j) ((ite (p i) (DFinsupp.singl …
  -/
  rw [filter_apply, this]
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    p : ι → Prop
    inst✝ : DecidablePred p
    i : ι
    x : β i
    j : ι
    this : Eq ((ite (p i) (DFinsupp.single i x) 0) j) (ite (p i) ((DFinsupp.single …
    ⊢ Eq (ite (p j) ((DFinsupp.single i x) j) 0) (ite (p i) ((DFinsupp.single i x) …
  -/
  obtain rfl | hij := Decidable.eq_or_ne i j
    /-
      case h.inl
      ι : Type u
      β : ι → Type v
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      i : ι
      x : β i
      this : Eq ((ite (p i) (DFinsupp.single i x) 0) i) (ite (p i) ((DFinsupp.single …
      ⊢ Eq (ite (p i) ((DFinsupp.single i x) i) 0) (ite (p i) ((DFinsupp.single i x) …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      β : ι → Type v
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : DecidableEq ι
      p : ι → Prop
      inst✝ : DecidablePred p
      i : ι
      x : β i
      j : ι
      this : Eq ((ite (p i) (DFinsupp.single i x) 0) j) (ite (p i) ((DFinsupp.single …
      hij : Ne i j
      ⊢ Eq (ite (p j) ((DFinsupp.single i x) j) 0) (ite (p i) ((DFinsupp.single i x) …
    -/
  · rw [single_eq_of_ne hij, ite_self, ite_self]
    /-
      🎉 no goals
    -/


@[simp]
theorem filter_single_pos {p : ι → Prop} [DecidablePred p] (i : ι) (x : β i) (h : p i) :
                                             /-
                                               ι : Type u
                                               β : ι → Type v
                                               inst✝² : (i : ι) → Zero (β i)
                                               inst✝¹ : DecidableEq ι
                                               p : ι → Prop
                                               inst✝ : DecidablePred p
                                               i : ι
                                               x : β i
                                               h : p i
                                               ⊢ Eq (DFinsupp.filter p (DFinsupp.single i x)) (DFinsupp.single i x)
                                             -/
    (single i x).filter p = single i x := by rw [filter_single, if_pos h]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem filter_single_neg {p : ι → Prop} [DecidablePred p] (i : ι) (x : β i) (h : ¬p i) :
                                    /-
                                      ι : Type u
                                      β : ι → Type v
                                      inst✝² : (i : ι) → Zero (β i)
                                      inst✝¹ : DecidableEq ι
                                      p : ι → Prop
                                      inst✝ : DecidablePred p
                                      i : ι
                                      x : β i
                                      h : Not (p i)
                                      ⊢ Eq (DFinsupp.filter p (DFinsupp.single i x)) 0
                                    -/
    (single i x).filter p = 0 := by rw [filter_single, if_neg h]
                                    /-
                                      🎉 no goals
                                    -/


/-- Equality of sigma types is sufficient (but not necessary) to show equality of `DFinsupp`s. -/
theorem single_eq_of_sigma_eq {i j} {xi : β i} {xj : β j} (h : (⟨i, xi⟩ : Sigma β) = ⟨j, xj⟩) :
    DFinsupp.single i xi = DFinsupp.single j xj := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i j : ι
    xi : β i
    xj : β j
    h : Eq ⟨i, xi⟩ ⟨j, xj⟩
    ⊢ Eq (DFinsupp.single i xi) (DFinsupp.single j xj)
  -/
  cases h
  /-
    case refl
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i : ι
    xi : β i
    ⊢ Eq (DFinsupp.single i xi) (DFinsupp.single i xi)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem equivFunOnFintype_single [Fintype ι] (i : ι) (m : β i) :
    (@DFinsupp.equivFunOnFintype ι β _ _) (DFinsupp.single i m) = Pi.single i m := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : β i
    ⊢ Eq (DFinsupp.equivFunOnFintype (DFinsupp.single i m)) (Pi.single i m)
  -/
  ext x
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : β i
    x : ι
    ⊢ Eq (DFinsupp.equivFunOnFintype (DFinsupp.single i m) x) (Pi.single i m x)
  -/
  dsimp [Pi.single, Function.update]
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : β i
    x : ι
    ⊢ Eq ((DFinsupp.single i m) x) (dite (Eq x i) (fun h => Eq.rec m ⋯) fun h => 0)
  -/
  simp [DFinsupp.single_eq_pi_single, @eq_comm _ i]
  /-
    🎉 no goals
  -/


@[simp]
theorem equivFunOnFintype_symm_single [Fintype ι] (i : ι) (m : β i) :
    (@DFinsupp.equivFunOnFintype ι β _ _).symm (Pi.single i m) = DFinsupp.single i m := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : β i
    ⊢ Eq (DFinsupp.equivFunOnFintype.symm (Pi.single i m)) (DFinsupp.single i m)
  -/
  ext i'
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    i : ι
    m : β i
    i' : ι
    ⊢ Eq ((DFinsupp.equivFunOnFintype.symm (Pi.single i m)) i') ((DFinsupp.single  …
  -/
  simp only [← single_eq_pi_single, equivFunOnFintype_symm_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem zipWith_single_single (f : ∀ i, β₁ i → β₂ i → β i) (hf : ∀ i, f i 0 0 = 0)
    {i} (b₁ : β₁ i) (b₂ : β₂ i) :
    zipWith f hf (single i b₁) (single i b₂) = single i (f i b₁ b₂) := by
  /-
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝³ : (i : ι) → Zero (β i)
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i → β i
    hf : ∀ (i : ι), Eq (f i 0 0) 0
    i : ι
    b₁ : β₁ i
    b₂ : β₂ i
    ⊢ Eq (DFinsupp.zipWith f hf (DFinsupp.single i b₁) (DFinsupp.single i b₂)) (DF …
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝³ : (i : ι) → Zero (β i)
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i → β i
    hf : ∀ (i : ι), Eq (f i 0 0) 0
    i : ι
    b₁ : β₁ i
    b₂ : β₂ i
    j : ι
    ⊢ Eq ((DFinsupp.zipWith f hf (DFinsupp.single i b₁) (DFinsupp.single i b₂)) j) …
  -/
  rw [zipWith_apply]
  /-
    case h
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝³ : (i : ι) → Zero (β i)
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β₁ i)
    inst✝ : (i : ι) → Zero (β₂ i)
    f : (i : ι) → β₁ i → β₂ i → β i
    hf : ∀ (i : ι), Eq (f i 0 0) 0
    i : ι
    b₁ : β₁ i
    b₂ : β₂ i
    j : ι
    ⊢ Eq (f j ((DFinsupp.single i b₁) j) ((DFinsupp.single i b₂) j)) ((DFinsupp.si …
  -/
  obtain rfl | hij := Decidable.eq_or_ne i j
    /-
      case h.inl
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝³ : (i : ι) → Zero (β i)
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      i : ι
      b₁ : β₁ i
      b₂ : β₂ i
      ⊢ Eq (f i ((DFinsupp.single i b₁) i) ((DFinsupp.single i b₂) i)) ((DFinsupp.si …
    -/
  · rw [single_eq_same, single_eq_same, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝³ : (i : ι) → Zero (β i)
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      i : ι
      b₁ : β₁ i
      b₂ : β₂ i
      j : ι
      hij : Ne i j
      ⊢ Eq (f j ((DFinsupp.single i b₁) j) ((DFinsupp.single i b₂) j)) ((DFinsupp.si …
    -/
  · rw [single_eq_of_ne hij, single_eq_of_ne hij, single_eq_of_ne hij, hf]
    /-
      🎉 no goals
    -/


/-- Redefine `f i` to be `0`. -/
def erase (i : ι) (x : Π₀ i, β i) : Π₀ i, β i :=
  ⟨fun j ↦ if j = i then 0 else x.1 j,
    x.support'.map fun xs ↦ ⟨xs.1, fun j ↦ (xs.prop j).imp_right (by simp only [·, ite_self])⟩⟩


@[simp]
theorem erase_apply {i j : ι} {f : Π₀ i, β i} : (f.erase i) j = if j = i then 0 else f j :=
  rfl


                                                                     /-
                                                                       ι : Type u
                                                                       β : ι → Type v
                                                                       inst✝¹ : (i : ι) → Zero (β i)
                                                                       inst✝ : DecidableEq ι
                                                                       i : ι
                                                                       f : DFinsupp fun i => β i
                                                                       ⊢ Eq ((DFinsupp.erase i f) i) 0
                                                                     -/
theorem erase_same {i : ι} {f : Π₀ i, β i} : (f.erase i) i = 0 := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                                       /-
                                                                                         ι : Type u
                                                                                         β : ι → Type v
                                                                                         inst✝¹ : (i : ι) → Zero (β i)
                                                                                         inst✝ : DecidableEq ι
                                                                                         i i' : ι
                                                                                         f : DFinsupp fun i => β i
                                                                                         h : Ne i' i
                                                                                         ⊢ Eq ((DFinsupp.erase i f) i') (f i')
                                                                                       -/
theorem erase_ne {i i' : ι} {f : Π₀ i, β i} (h : i' ≠ i) : (f.erase i) i' = f i' := by simp [h]
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem piecewise_single_erase (x : Π₀ i, β i) (i : ι)
    [∀ i' : ι, Decidable <| (i' ∈ ({i} : Set ι))] : -- Porting note: added Decidable hypothesis
    (single i (x i)).piecewise (x.erase i) {i} = x := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : DecidableEq ι
    x : DFinsupp fun i => β i
    i : ι
    inst✝ : (i' : ι) → Decidable (Membership.mem (Singleton.singleton i) i')
    ⊢ Eq ((DFinsupp.single i (x i)).piecewise (DFinsupp.erase i x) (Singleton.sing …
  -/
  ext j; rw [piecewise_apply]; split_ifs with h
    /-
      case pos
      ι : Type u
      β : ι → Type v
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : DecidableEq ι
      x : DFinsupp fun i => β i
      i : ι
      inst✝ : (i' : ι) → Decidable (Membership.mem (Singleton.singleton i) i')
      j : ι
      h : Membership.mem (Singleton.singleton i) j
      ⊢ Eq ((DFinsupp.single i (x i)) j) (x j)
    -/
  · rw [(id h : j = i), single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      β : ι → Type v
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : DecidableEq ι
      x : DFinsupp fun i => β i
      i : ι
      inst✝ : (i' : ι) → Decidable (Membership.mem (Singleton.singleton i) i')
      j : ι
      h : Not (Membership.mem (Singleton.singleton i) j)
      ⊢ Eq ((DFinsupp.erase i x) j) (x j)
    -/
  · exact erase_ne h
    /-
      🎉 no goals
    -/


theorem erase_eq_sub_single {β : ι → Type*} [∀ i, AddGroup (β i)] (f : Π₀ i, β i) (i : ι) :
    f.erase i = f - single i (f i) := by
  /-
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddGroup (β i)
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Eq (DFinsupp.erase i f) (HSub.hSub f (DFinsupp.single i (f i)))
  -/
  ext j
  /-
    case h
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddGroup (β i)
    f : DFinsupp fun i => β i
    i j : ι
    ⊢ Eq ((DFinsupp.erase i f) j) ((HSub.hSub f (DFinsupp.single i (f i))) j)
  -/
  rcases eq_or_ne i j with (rfl | h)
    /-
      case h.inl
      ι : Type u
      inst✝¹ : DecidableEq ι
      β : ι → Type u_1
      inst✝ : (i : ι) → AddGroup (β i)
      f : DFinsupp fun i => β i
      i : ι
      ⊢ Eq ((DFinsupp.erase i f) i) ((HSub.hSub f (DFinsupp.single i (f i))) i)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      inst✝¹ : DecidableEq ι
      β : ι → Type u_1
      inst✝ : (i : ι) → AddGroup (β i)
      f : DFinsupp fun i => β i
      i j : ι
      h : Ne i j
      ⊢ Eq ((DFinsupp.erase i f) j) ((HSub.hSub f (DFinsupp.single i (f i))) j)
    -/
  · simp [erase_ne h.symm, single_eq_of_ne h, @eq_comm _ j, h]
    /-
      🎉 no goals
    -/


@[simp]
theorem erase_zero (i : ι) : erase i (0 : Π₀ i, β i) = 0 :=
  ext fun _ => ite_self _


@[simp]
theorem filter_ne_eq_erase (f : Π₀ i, β i) (i : ι) : f.filter (· ≠ i) = f.erase i := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Eq (DFinsupp.filter (fun x => Ne x i) f) (DFinsupp.erase i f)
  -/
  ext1 j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i j : ι
    ⊢ Eq ((DFinsupp.filter (fun x => Ne x i) f) j) ((DFinsupp.erase i f) j)
  -/
  simp only [DFinsupp.filter_apply, DFinsupp.erase_apply, ite_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem filter_ne_eq_erase' (f : Π₀ i, β i) (i : ι) : f.filter (i ≠ ·) = f.erase i := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Eq (DFinsupp.filter (fun x => Ne i x) f) (DFinsupp.erase i f)
  -/
  rw [← filter_ne_eq_erase f i]
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Eq (DFinsupp.filter (fun x => Ne i x) f) (DFinsupp.filter (fun x => Ne x i) f)
  -/
  congr with j
  /-
    case e_p.h.a
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i j : ι
    ⊢ Iff (Ne i j) (Ne j i)
  -/
  exact ne_comm
  /-
    🎉 no goals
  -/


theorem erase_single (j : ι) (i : ι) (x : β i) :
    (single i x).erase j = if i = j then 0 else single i x := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    j i : ι
    x : β i
    ⊢ Eq (DFinsupp.erase j (DFinsupp.single i x)) (ite (Eq i j) 0 (DFinsupp.single …
  -/
  rw [← filter_ne_eq_erase, filter_single, ite_not]
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_single_same (i : ι) (x : β i) : (single i x).erase i = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i : ι
    x : β i
    ⊢ Eq (DFinsupp.erase i (DFinsupp.single i x)) 0
  -/
  rw [erase_single, if_pos rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_single_ne {i j : ι} (x : β i) (h : i ≠ j) : (single i x).erase j = single i x := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    i j : ι
    x : β i
    h : Ne i j
    ⊢ Eq (DFinsupp.erase j (DFinsupp.single i x)) (DFinsupp.single i x)
  -/
  rw [erase_single, if_neg h]
  /-
    🎉 no goals
  -/


/-- Replace the value of a `Π₀ i, β i` at a given point `i : ι` by a given value `b : β i`.
If `b = 0`, this amounts to removing `i` from the support.
Otherwise, `i` is added to it.

This is the (dependent) finitely-supported version of `Function.update`. -/
def update : Π₀ i, β i :=
  ⟨Function.update f i b,
    f.support'.map fun s =>
      ⟨i ::ₘ s.1, fun j => by
        /-
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝¹ : (i : ι) → Zero (β i)
          inst✝ : DecidableEq ι
          f : DFinsupp fun i => β i
          i : ι
          b : β i
          s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
          j : ι
          ⊢ Or (Membership.mem (Multiset.cons i ↑s) j) (Eq (Function.update (⇑f) i b j) 0)
        -/
        rcases eq_or_ne i j with (rfl | hi)
          /-
            case inl
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝¹ : (i : ι) → Zero (β i)
            inst✝ : DecidableEq ι
            f : DFinsupp fun i => β i
            i : ι
            b : β i
            s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
            ⊢ Or (Membership.mem (Multiset.cons i ↑s) i) (Eq (Function.update (⇑f) i b i) 0)
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case inr
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝¹ : (i : ι) → Zero (β i)
            inst✝ : DecidableEq ι
            f : DFinsupp fun i => β i
            i : ι
            b : β i
            s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
            j : ι
            hi : Ne i j
            ⊢ Or (Membership.mem (Multiset.cons i ↑s) j) (Eq (Function.update (⇑f) i b j) 0)
          -/
        · obtain hj | (hj : f j = 0) := s.prop j
            /-
              case inr.inl
              ι : Type u
              γ : Type w
              β : ι → Type v
              β₁ : ι → Type v₁
              β₂ : ι → Type v₂
              inst✝¹ : (i : ι) → Zero (β i)
              inst✝ : DecidableEq ι
              f : DFinsupp fun i => β i
              i : ι
              b : β i
              s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
              j : ι
              hi : Ne i j
              hj : Membership.mem (↑s) j
              ⊢ Or (Membership.mem (Multiset.cons i ↑s) j) (Eq (Function.update (⇑f) i b j) 0)
            -/
          · exact Or.inl (Multiset.mem_cons_of_mem hj)
            /-
              🎉 no goals
            -/
            /-
              case inr.inr
              ι : Type u
              γ : Type w
              β : ι → Type v
              β₁ : ι → Type v₁
              β₂ : ι → Type v₂
              inst✝¹ : (i : ι) → Zero (β i)
              inst✝ : DecidableEq ι
              f : DFinsupp fun i => β i
              i : ι
              b : β i
              s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
              j : ι
              hi : Ne i j
              hj : Eq (f j) 0
              ⊢ Or (Membership.mem (Multiset.cons i ↑s) j) (Eq (Function.update (⇑f) i b j) 0)
            -/
          · exact Or.inr ((Function.update_of_ne hi.symm b _).trans hj)⟩⟩
            /-
              🎉 no goals
            -/


@[simp, norm_cast] lemma coe_update : (f.update i b : ∀ i : ι, β i) = Function.update f i b := rfl


@[simp]
theorem update_self : f.update i (f i) = f := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Eq (f.update i (f i)) f
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i i✝ : ι
    ⊢ Eq ((f.update i (f i)) i✝) (f i✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem update_eq_erase : f.update i 0 = f.erase i := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Eq (f.update i 0) (DFinsupp.erase i f)
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : DecidableEq ι
    f : DFinsupp fun i => β i
    i j : ι
    ⊢ Eq ((f.update i 0) j) ((DFinsupp.erase i f) j)
  -/
  rcases eq_or_ne i j with (rfl | hi)
    /-
      case h.inl
      ι : Type u
      β : ι → Type v
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : DecidableEq ι
      f : DFinsupp fun i => β i
      i : ι
      ⊢ Eq ((f.update i 0) i) ((DFinsupp.erase i f) i)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      β : ι → Type v
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : DecidableEq ι
      f : DFinsupp fun i => β i
      i j : ι
      hi : Ne i j
      ⊢ Eq ((f.update i 0) j) ((DFinsupp.erase i f) j)
    -/
  · simp [hi.symm]
    /-
      🎉 no goals
    -/


theorem update_eq_single_add_erase {β : ι → Type*} [∀ i, AddZeroClass (β i)] (f : Π₀ i, β i)
    (i : ι) (b : β i) : f.update i b = single i b + f.erase i := by
  /-
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    ⊢ Eq (f.update i b) (HAdd.hAdd (DFinsupp.single i b) (DFinsupp.erase i f))
  -/
  ext j
  /-
    case h
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    j : ι
    ⊢ Eq ((f.update i b) j) ((HAdd.hAdd (DFinsupp.single i b) (DFinsupp.erase i f) …
  -/
  rcases eq_or_ne i j with (rfl | h)
    /-
      case h.inl
      ι : Type u
      inst✝¹ : DecidableEq ι
      β : ι → Type u_1
      inst✝ : (i : ι) → AddZeroClass (β i)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      ⊢ Eq ((f.update i b) i) ((HAdd.hAdd (DFinsupp.single i b) (DFinsupp.erase i f) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      inst✝¹ : DecidableEq ι
      β : ι → Type u_1
      inst✝ : (i : ι) → AddZeroClass (β i)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      j : ι
      h : Ne i j
      ⊢ Eq ((f.update i b) j) ((HAdd.hAdd (DFinsupp.single i b) (DFinsupp.erase i f) …
    -/
  · simp [Function.update_of_ne h.symm, h, erase_ne, h.symm]
    /-
      🎉 no goals
    -/


theorem update_eq_erase_add_single {β : ι → Type*} [∀ i, AddZeroClass (β i)] (f : Π₀ i, β i)
    (i : ι) (b : β i) : f.update i b = f.erase i + single i b := by
  /-
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    ⊢ Eq (f.update i b) (HAdd.hAdd (DFinsupp.erase i f) (DFinsupp.single i b))
  -/
  ext j
  /-
    case h
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    j : ι
    ⊢ Eq ((f.update i b) j) ((HAdd.hAdd (DFinsupp.erase i f) (DFinsupp.single i b) …
  -/
  rcases eq_or_ne i j with (rfl | h)
    /-
      case h.inl
      ι : Type u
      inst✝¹ : DecidableEq ι
      β : ι → Type u_1
      inst✝ : (i : ι) → AddZeroClass (β i)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      ⊢ Eq ((f.update i b) i) ((HAdd.hAdd (DFinsupp.erase i f) (DFinsupp.single i b) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      inst✝¹ : DecidableEq ι
      β : ι → Type u_1
      inst✝ : (i : ι) → AddZeroClass (β i)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      j : ι
      h : Ne i j
      ⊢ Eq ((f.update i b) j) ((HAdd.hAdd (DFinsupp.erase i f) (DFinsupp.single i b) …
    -/
  · simp [Function.update_of_ne h.symm, h, erase_ne, h.symm]
    /-
      🎉 no goals
    -/


theorem update_eq_sub_add_single {β : ι → Type*} [∀ i, AddGroup (β i)] (f : Π₀ i, β i) (i : ι)
    (b : β i) : f.update i b = f - single i (f i) + single i b := by
  /-
    ι : Type u
    inst✝¹ : DecidableEq ι
    β : ι → Type u_1
    inst✝ : (i : ι) → AddGroup (β i)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    ⊢ Eq (f.update i b) (HAdd.hAdd (HSub.hSub f (DFinsupp.single i (f i))) (DFinsu …
  -/
  rw [update_eq_erase_add_single f i b, erase_eq_sub_single f i]
  /-
    🎉 no goals
  -/


@[simp]
theorem single_add (i : ι) (b₁ b₂ : β i) : single i (b₁ + b₂) = single i b₁ + single i b₂ :=
  (zipWith_single_single (fun _ => (· + ·)) _ b₁ b₂).symm


@[simp]
theorem erase_add (i : ι) (f₁ f₂ : Π₀ i, β i) : erase i (f₁ + f₂) = erase i f₁ + erase i f₂ :=
                  /-
                    ι : Type u
                    β : ι → Type v
                    inst✝¹ : DecidableEq ι
                    inst✝ : (i : ι) → AddZeroClass (β i)
                    i : ι
                    f₁ f₂ : DFinsupp fun i => β i
                    x✝ : ι
                    ⊢ Eq ((DFinsupp.erase i (HAdd.hAdd f₁ f₂)) x✝) ((HAdd.hAdd (DFinsupp.erase i f …
                  -/
  ext fun _ => by simp [ite_zero_add]
                  /-
                    🎉 no goals
                  -/


/-- `DFinsupp.single` as an `AddMonoidHom`. -/
@[simps]
def singleAddHom (i : ι) : β i →+ Π₀ i, β i where
  toFun := single i
  map_zero' := single_zero i
  map_add' := single_add i


/-- `DFinsupp.erase` as an `AddMonoidHom`. -/
@[simps]
def eraseAddHom (i : ι) : (Π₀ i, β i) →+ Π₀ i, β i where
  toFun := erase i
  map_zero' := erase_zero i
  map_add' := erase_add i


@[simp]
theorem single_neg {β : ι → Type v} [∀ i, AddGroup (β i)] (i : ι) (x : β i) :
    single i (-x) = -single i x :=
  (singleAddHom β i).map_neg x


@[simp]
theorem single_sub {β : ι → Type v} [∀ i, AddGroup (β i)] (i : ι) (x y : β i) :
    single i (x - y) = single i x - single i y :=
  (singleAddHom β i).map_sub x y


@[simp]
theorem erase_neg {β : ι → Type v} [∀ i, AddGroup (β i)] (i : ι) (f : Π₀ i, β i) :
    (-f).erase i = -f.erase i :=
  (eraseAddHom β i).map_neg f


@[simp]
theorem erase_sub {β : ι → Type v} [∀ i, AddGroup (β i)] (i : ι) (f g : Π₀ i, β i) :
    (f - g).erase i = f.erase i - g.erase i :=
  (eraseAddHom β i).map_sub f g


theorem single_add_erase (i : ι) (f : Π₀ i, β i) : single i (f i) + f.erase i = f :=
  ext fun i' =>
    if h : i = i' then by
      /-
        ι : Type u
        β : ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → AddZeroClass (β i)
        i : ι
        f : DFinsupp fun i => β i
        i' : ι
        h : Eq i i'
        ⊢ Eq ((HAdd.hAdd (DFinsupp.single i (f i)) (DFinsupp.erase i f)) i') (f i')
      -/
      subst h; simp only [add_apply, single_apply, erase_apply, add_zero, dite_eq_ite, if_true]
               /-
                 🎉 no goals
               -/
    else by
      /-
        ι : Type u
        β : ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → AddZeroClass (β i)
        i : ι
        f : DFinsupp fun i => β i
        i' : ι
        h : Not (Eq i i')
        ⊢ Eq ((HAdd.hAdd (DFinsupp.single i (f i)) (DFinsupp.erase i f)) i') (f i')
      -/
      simp only [add_apply, single_apply, erase_apply, dif_neg h, if_neg (Ne.symm h), zero_add]
      /-
        🎉 no goals
      -/


theorem erase_add_single (i : ι) (f : Π₀ i, β i) : f.erase i + single i (f i) = f :=
  ext fun i' =>
    if h : i = i' then by
      /-
        ι : Type u
        β : ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → AddZeroClass (β i)
        i : ι
        f : DFinsupp fun i => β i
        i' : ι
        h : Eq i i'
        ⊢ Eq ((HAdd.hAdd (DFinsupp.erase i f) (DFinsupp.single i (f i))) i') (f i')
      -/
      subst h; simp only [add_apply, single_apply, erase_apply, zero_add, dite_eq_ite, if_true]
               /-
                 🎉 no goals
               -/
    else by
      /-
        ι : Type u
        β : ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → AddZeroClass (β i)
        i : ι
        f : DFinsupp fun i => β i
        i' : ι
        h : Not (Eq i i')
        ⊢ Eq ((HAdd.hAdd (DFinsupp.erase i f) (DFinsupp.single i (f i))) i') (f i')
      -/
      simp only [add_apply, single_apply, erase_apply, dif_neg h, if_neg (Ne.symm h), add_zero]
      /-
        🎉 no goals
      -/


protected theorem induction {p : (Π₀ i, β i) → Prop} (f : Π₀ i, β i) (h0 : p 0)
    (ha : ∀ (i b) (f : Π₀ i, β i), f i = 0 → b ≠ 0 → p f → p (single i b + f)) : p f := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    f : DFinsupp fun i => β i
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    ⊢ p f
  -/
  cases' f with f s
  /-
    case mk'
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    f : (i : ι) → β i
    s : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0))
    ⊢ p { toFun := f, support' := s }
  -/
  induction' s using Trunc.induction_on with s
  /-
    case mk'.h
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    f : (i : ι) → β i
    s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ p { toFun := f, support' := Trunc.mk s }
  -/
  cases' s with s H
  /-
    case mk'.h.mk
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    f : (i : ι) → β i
    s : Multiset ι
    H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ p { toFun := f, support' := Trunc.mk ⟨s, H⟩ }
  -/
  induction' s using Multiset.induction_on with i s ih generalizing f
    /-
      case mk'.h.mk.empty
      ι : Type u
      β : ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → AddZeroClass (β i)
      p : (DFinsupp fun i => β i) → Prop
      h0 : p 0
      ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
      f : (i : ι) → β i
      H : ∀ (i : ι), Or (Membership.mem 0 i) (Eq (f i) 0)
      ⊢ p { toFun := f, support' := Trunc.mk ⟨0, H⟩ }
    -/
  · have : f = 0 := funext fun i => (H i).resolve_left (Multiset.not_mem_zero _)
    /-
      case mk'.h.mk.empty
      ι : Type u
      β : ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → AddZeroClass (β i)
      p : (DFinsupp fun i => β i) → Prop
      h0 : p 0
      ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
      f : (i : ι) → β i
      H : ∀ (i : ι), Or (Membership.mem 0 i) (Eq (f i) 0)
      this : Eq f 0
      ⊢ p { toFun := f, support' := Trunc.mk ⟨0, H⟩ }
    -/
    subst this
    /-
      case mk'.h.mk.empty
      ι : Type u
      β : ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → AddZeroClass (β i)
      p : (DFinsupp fun i => β i) → Prop
      h0 : p 0
      ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
      H : ∀ (i : ι), Or (Membership.mem 0 i) (Eq (0 i) 0)
      ⊢ p { toFun := 0, support' := Trunc.mk ⟨0, H⟩ }
    -/
    exact h0
    /-
      🎉 no goals
    -/
  have H2 : p (erase i ⟨f, Trunc.mk ⟨i ::ₘ s, H⟩⟩) := by
    dsimp only [erase, Trunc.map, Trunc.bind, Trunc.liftOn, Trunc.lift_mk,
      Function.comp, Subtype.coe_mk]
    have H2 : ∀ j, j ∈ s ∨ ite (j = i) 0 (f j) = 0 := by
      intro j
      cases' H j with H2 H2
      · cases' Multiset.mem_cons.1 H2 with H3 H3
        · right; exact if_pos H3
        · left; exact H3
      right
      split_ifs <;> [rfl; exact H2]
    have H3 : ∀ aux, (⟨fun j : ι => ite (j = i) 0 (f j), Trunc.mk ⟨i ::ₘ s, aux⟩⟩ : Π₀ i, β i) =
        ⟨fun j : ι => ite (j = i) 0 (f j), Trunc.mk ⟨s, H2⟩⟩ :=
      fun _ ↦ ext fun _ => rfl
    rw [H3]
    apply ih
  /-
    case mk'.h.mk.cons
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    i : ι
    s : Multiset ι
    ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
    f : (i : ι) → β i
    H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
    H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
    ⊢ p { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s, H⟩ }
  -/
  have H3 : single i _ + _ = (⟨f, Trunc.mk ⟨i ::ₘ s, H⟩⟩ : Π₀ i, β i) := single_add_erase _ _
  /-
    case mk'.h.mk.cons
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    i : ι
    s : Multiset ι
    ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
    f : (i : ι) → β i
    H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
    H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
    H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
    ⊢ p { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s, H⟩ }
  -/
  rw [← H3]
  /-
    case mk'.h.mk.cons
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    i : ι
    s : Multiset ι
    ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
    f : (i : ι) → β i
    H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
    H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
    H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
    ⊢ p (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Multise …
  -/
  change p (single i (f i) + _)
  /-
    case mk'.h.mk.cons
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    i : ι
    s : Multiset ι
    ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
    f : (i : ι) → β i
    H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
    H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
    H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
    ⊢ p (HAdd.hAdd (DFinsupp.single i (f i)) (DFinsupp.erase i { toFun := f, suppo …
  -/
  cases' Classical.em (f i = 0) with h h
    /-
      case mk'.h.mk.cons.inl
      ι : Type u
      β : ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → AddZeroClass (β i)
      p : (DFinsupp fun i => β i) → Prop
      h0 : p 0
      ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
      i : ι
      s : Multiset ι
      ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
      f : (i : ι) → β i
      H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
      H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
      H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
      h : Eq (f i) 0
      ⊢ p (HAdd.hAdd (DFinsupp.single i (f i)) (DFinsupp.erase i { toFun := f, suppo …
    -/
  · rw [h, single_zero, zero_add]
    /-
      case mk'.h.mk.cons.inl
      ι : Type u
      β : ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → AddZeroClass (β i)
      p : (DFinsupp fun i => β i) → Prop
      h0 : p 0
      ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
      i : ι
      s : Multiset ι
      ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
      f : (i : ι) → β i
      H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
      H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
      H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
      h : Eq (f i) 0
      ⊢ p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s, H …
    -/
    exact H2
    /-
      🎉 no goals
    -/
  /-
    case mk'.h.mk.cons.inr
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    i : ι
    s : Multiset ι
    ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
    f : (i : ι) → β i
    H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
    H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
    H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
    h : Not (Eq (f i) 0)
    ⊢ p (HAdd.hAdd (DFinsupp.single i (f i)) (DFinsupp.erase i { toFun := f, suppo …
  -/
  refine ha _ _ _ ?_ h H2
  /-
    case mk'.h.mk.cons.inr
    ι : Type u
    β : ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → AddZeroClass (β i)
    p : (DFinsupp fun i => β i) → Prop
    h0 : p 0
    ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
    i : ι
    s : Multiset ι
    ih : ∀ (f : (i : ι) → β i) (H : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0 …
    f : (i : ι) → β i
    H : ∀ (i_1 : ι), Or (Membership.mem (Multiset.cons i s) i_1) (Eq (f i_1) 0)
    H2 : p (DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s …
    H3 : Eq (HAdd.hAdd (DFinsupp.single i ({ toFun := f, support' := Trunc.mk ⟨Mul …
    h : Not (Eq (f i) 0)
    ⊢ Eq ((DFinsupp.erase i { toFun := f, support' := Trunc.mk ⟨Multiset.cons i s, …
  -/
  rw [erase_same]
  /-
    🎉 no goals
  -/


theorem induction₂ {p : (Π₀ i, β i) → Prop} (f : Π₀ i, β i) (h0 : p 0)
    (ha : ∀ (i b) (f : Π₀ i, β i), f i = 0 → b ≠ 0 → p f → p (f + single i b)) : p f :=
  DFinsupp.induction f h0 fun i b f h1 h2 h3 =>
    have h4 : f + single i b = single i b + f := by
      /-
        ι : Type u
        β : ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : ι) → AddZeroClass (β i)
        p : (DFinsupp fun i => β i) → Prop
        f✝ : DFinsupp fun i => β i
        h0 : p 0
        ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
        i : ι
        b : β i
        f : DFinsupp fun i => β i
        h1 : Eq (f i) 0
        h2 : Ne b 0
        h3 : p f
        ⊢ Eq (HAdd.hAdd f (DFinsupp.single i b)) (HAdd.hAdd (DFinsupp.single i b) f)
      -/
      ext j; by_cases H : i = j
        /-
          case pos
          ι : Type u
          β : ι → Type v
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → AddZeroClass (β i)
          p : (DFinsupp fun i => β i) → Prop
          f✝ : DFinsupp fun i => β i
          h0 : p 0
          ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
          i : ι
          b : β i
          f : DFinsupp fun i => β i
          h1 : Eq (f i) 0
          h2 : Ne b 0
          h3 : p f
          j : ι
          H : Eq i j
          ⊢ Eq ((HAdd.hAdd f (DFinsupp.single i b)) j) ((HAdd.hAdd (DFinsupp.single i b) …
        -/
      · subst H
        /-
          case pos
          ι : Type u
          β : ι → Type v
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → AddZeroClass (β i)
          p : (DFinsupp fun i => β i) → Prop
          f✝ : DFinsupp fun i => β i
          h0 : p 0
          ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
          i : ι
          b : β i
          f : DFinsupp fun i => β i
          h1 : Eq (f i) 0
          h2 : Ne b 0
          h3 : p f
          ⊢ Eq ((HAdd.hAdd f (DFinsupp.single i b)) i) ((HAdd.hAdd (DFinsupp.single i b) …
        -/
        simp [h1]
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u
          β : ι → Type v
          inst✝¹ : DecidableEq ι
          inst✝ : (i : ι) → AddZeroClass (β i)
          p : (DFinsupp fun i => β i) → Prop
          f✝ : DFinsupp fun i => β i
          h0 : p 0
          ha : ∀ (i : ι) (b : β i) (f : DFinsupp fun i => β i), Eq (f i) 0 → Ne b 0 → p  …
          i : ι
          b : β i
          f : DFinsupp fun i => β i
          h1 : Eq (f i) 0
          h2 : Ne b 0
          h3 : p f
          j : ι
          H : Not (Eq i j)
          ⊢ Eq ((HAdd.hAdd f (DFinsupp.single i b)) j) ((HAdd.hAdd (DFinsupp.single i b) …
        -/
      · simp [H]
        /-
          🎉 no goals
        -/
    Eq.recOn h4 <| ha i b f h1 h2 h3


@[simp]
theorem mk_add [∀ i, AddZeroClass (β i)] {s : Finset ι} {x y : ∀ i : (↑s : Set ι), β i} :
    mk s (x + y) = mk s x + mk s y :=
                  /-
                    ι : Type u
                    β : ι → Type v
                    inst✝¹ : DecidableEq ι
                    inst✝ : (i : ι) → AddZeroClass (β i)
                    s : Finset ι
                    x y : (i : ↑↑s) → β ↑i
                    i : ι
                    ⊢ Eq ((DFinsupp.mk s (HAdd.hAdd x y)) i) ((HAdd.hAdd (DFinsupp.mk s x) (DFinsu …
                  -/
  ext fun i => by simp only [add_apply, mk_apply]; split_ifs <;> [rfl; rw [zero_add]]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem mk_zero [∀ i, Zero (β i)] {s : Finset ι} : mk s (0 : ∀ i : (↑s : Set ι), β i.1) = 0 :=
                  /-
                    ι : Type u
                    β : ι → Type v
                    inst✝¹ : DecidableEq ι
                    inst✝ : (i : ι) → Zero (β i)
                    s : Finset ι
                    i : ι
                    ⊢ Eq ((DFinsupp.mk s 0) i) (0 i)
                  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  ext fun i => by simp only [mk_apply]; split_ifs <;> rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem mk_neg [∀ i, AddGroup (β i)] {s : Finset ι} {x : ∀ i : (↑s : Set ι), β i.1} :
    mk s (-x) = -mk s x :=
                  /-
                    ι : Type u
                    β : ι → Type v
                    inst✝¹ : DecidableEq ι
                    inst✝ : (i : ι) → AddGroup (β i)
                    s : Finset ι
                    x : (i : ↑↑s) → β ↑i
                    i : ι
                    ⊢ Eq ((DFinsupp.mk s (Neg.neg x)) i) ((Neg.neg (DFinsupp.mk s x)) i)
                  -/
  ext fun i => by simp only [neg_apply, mk_apply]; split_ifs <;> [rfl; rw [neg_zero]]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem mk_sub [∀ i, AddGroup (β i)] {s : Finset ι} {x y : ∀ i : (↑s : Set ι), β i.1} :
    mk s (x - y) = mk s x - mk s y :=
                  /-
                    ι : Type u
                    β : ι → Type v
                    inst✝¹ : DecidableEq ι
                    inst✝ : (i : ι) → AddGroup (β i)
                    s : Finset ι
                    x y : (i : ↑↑s) → β ↑i
                    i : ι
                    ⊢ Eq ((DFinsupp.mk s (HSub.hSub x y)) i) ((HSub.hSub (DFinsupp.mk s x) (DFinsu …
                  -/
  ext fun i => by simp only [sub_apply, mk_apply]; split_ifs <;> [rfl; rw [sub_zero]]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If `s` is a subset of `ι` then `mk_addGroupHom s` is the canonical additive
group homomorphism from $\prod_{i\in s}\beta_i$ to $\prod_{\mathtt{i : \iota}}\beta_i.$-/
def mkAddGroupHom [∀ i, AddGroup (β i)] (s : Finset ι) :
    (∀ i : (s : Set ι), β ↑i) →+ Π₀ i : ι, β i where
  toFun := mk s
  map_zero' := mk_zero
  map_add' _ _ := mk_add


/-- Set `{i | f x ≠ 0}` as a `Finset`. -/
def support (f : Π₀ i, β i) : Finset ι :=
  (f.support'.lift fun xs => (Multiset.toFinset xs.1).filter fun i => f i ≠ 0) <| by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      ⊢ ∀ (a b : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) …
    -/
    rintro ⟨sx, hx⟩ ⟨sy, hy⟩
    /-
      case mk.mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      sx : Multiset ι
      hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
      sy : Multiset ι
      hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
      ⊢ Eq ((fun xs => Finset.filter (fun i => Ne (f i) 0) (↑xs).toFinset) ⟨sx, hx⟩) …
    -/
    dsimp only [Subtype.coe_mk, toFun_eq_coe] at *
    /-
      case mk.mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      sx : Multiset ι
      hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
      sy : Multiset ι
      hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
      ⊢ Eq (Finset.filter (fun i => Ne (f i) 0) sx.toFinset) (Finset.filter (fun i = …
    -/
    ext i; constructor
      /-
        case mk.mk.h.mp
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        i : ι
        ⊢ Membership.mem (Finset.filter (fun i => Ne (f i) 0) sx.toFinset) i → Members …
      -/
    · intro H
      /-
        case mk.mk.h.mp
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        i : ι
        H : Membership.mem (Finset.filter (fun i => Ne (f i) 0) sx.toFinset) i
        ⊢ Membership.mem (Finset.filter (fun i => Ne (f i) 0) sy.toFinset) i
      -/
      rcases Finset.mem_filter.1 H with ⟨_, h⟩
      /-
        case mk.mk.h.mp.intro
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        i : ι
        H : Membership.mem (Finset.filter (fun i => Ne (f i) 0) sx.toFinset) i
        left✝ : Membership.mem sx.toFinset i
        h : Ne (f i) 0
        ⊢ Membership.mem (Finset.filter (fun i => Ne (f i) 0) sy.toFinset) i
      -/
      exact Finset.mem_filter.2 ⟨Multiset.mem_toFinset.2 <| (hy i).resolve_right h, h⟩
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.h.mpr
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        i : ι
        ⊢ Membership.mem (Finset.filter (fun i => Ne (f i) 0) sy.toFinset) i → Members …
      -/
    · intro H
      /-
        case mk.mk.h.mpr
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        i : ι
        H : Membership.mem (Finset.filter (fun i => Ne (f i) 0) sy.toFinset) i
        ⊢ Membership.mem (Finset.filter (fun i => Ne (f i) 0) sx.toFinset) i
      -/
      rcases Finset.mem_filter.1 H with ⟨_, h⟩
      /-
        case mk.mk.h.mpr.intro
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        f : DFinsupp fun i => β i
        sx : Multiset ι
        hx : ∀ (i : ι), Or (Membership.mem sx i) (Eq (f.toFun i) 0)
        sy : Multiset ι
        hy : ∀ (i : ι), Or (Membership.mem sy i) (Eq (f.toFun i) 0)
        i : ι
        H : Membership.mem (Finset.filter (fun i => Ne (f i) 0) sy.toFinset) i
        left✝ : Membership.mem sy.toFinset i
        h : Ne (f i) 0
        ⊢ Membership.mem (Finset.filter (fun i => Ne (f i) 0) sx.toFinset) i
      -/
      exact Finset.mem_filter.2 ⟨Multiset.mem_toFinset.2 <| (hx i).resolve_right h, h⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem support_mk_subset {s : Finset ι} {x : ∀ i : (↑s : Set ι), β i.1} : (mk s x).support ⊆ s :=
  fun _ H => Multiset.mem_toFinset.1 (Finset.mem_filter.1 H).1


@[simp]
theorem support_mk'_subset {f : ∀ i, β i} {s : Multiset ι} {h} :
    (mk' f <| Trunc.mk ⟨s, h⟩).support ⊆ s.toFinset := fun i H =>
                                /-
                                  ι : Type u
                                  β : ι → Type v
                                  inst✝² : DecidableEq ι
                                  inst✝¹ : (i : ι) → Zero (β i)
                                  inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
                                  f : (i : ι) → β i
                                  s : Multiset ι
                                  h : ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
                                  i : ι
                                  H : Membership.mem { toFun := f, support' := Trunc.mk ⟨s, h⟩ }.support i
                                  ⊢ Membership.mem s.toFinset.val.toFinset i
                                -/
  Multiset.mem_toFinset.1 <| by simpa using (Finset.mem_filter.1 H).1
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem mem_support_toFun (f : Π₀ i, β i) (i) : i ∈ f.support ↔ f i ≠ 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    i : ι
    ⊢ Iff (Membership.mem f.support i) (Ne (f i) 0)
  -/
  cases' f with f s
  /-
    case mk'
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : (i : ι) → β i
    s : Trunc (Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0))
    ⊢ Iff (Membership.mem { toFun := f, support' := s }.support i) (Ne ({ toFun := …
  -/
  induction' s using Trunc.induction_on with s
  /-
    case mk'.h
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : (i : ι) → β i
    s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ Iff (Membership.mem { toFun := f, support' := Trunc.mk s }.support i) (Ne ({ …
  -/
  dsimp only [support, Trunc.lift_mk]
  /-
    case mk'.h
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : (i : ι) → β i
    s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ Iff (Membership.mem (Finset.filter (fun i => Ne ({ toFun := f, support' := T …
  -/
  rw [Finset.mem_filter, Multiset.mem_toFinset, coe_mk']
  /-
    case mk'.h
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : (i : ι) → β i
    s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f i) 0)
    ⊢ Iff (And (Membership.mem (↑s) i) (Ne (f i) 0)) (Ne (f i) 0)
  -/
  exact and_iff_right_of_imp (s.prop i).resolve_right
  /-
    🎉 no goals
  -/


                                                                            /-
                                                                              ι : Type u
                                                                              β : ι → Type v
                                                                              inst✝² : DecidableEq ι
                                                                              inst✝¹ : (i : ι) → Zero (β i)
                                                                              inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
                                                                              f : DFinsupp fun i => β i
                                                                              ⊢ Eq f (DFinsupp.mk f.support fun i => f ↑i)
                                                                            -/
theorem eq_mk_support (f : Π₀ i, β i) : f = mk f.support fun i => f i := by aesop
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- Equivalence between dependent functions with finite support `s : Finset ι` and functions
`∀ i, {x : β i // x ≠ 0}`. -/
@[simps]
def subtypeSupportEqEquiv (s : Finset ι) :
    {f : Π₀ i, β i // f.support = s} ≃ ∀ i : s, {x : β i // x ≠ 0} where
  toFun | ⟨f, hf⟩ => fun ⟨i, hi⟩ ↦ ⟨f i, (f.mem_support_toFun i).1 <| hf.symm ▸ hi⟩
  invFun f := ⟨mk s fun i ↦ (f i).1, Finset.ext fun i ↦ by
    -- TODO: `simp` fails to use `(f _).2` inside `∃ _, _`
    calc
      i ∈ support (mk s fun i ↦ (f i).1) ↔ ∃ h : i ∈ s, (f ⟨i, h⟩).1 ≠ 0 := by simp
      _ ↔ ∃ _ : i ∈ s, True := exists_congr fun h ↦ (iff_true _).mpr (f _).2
      _ ↔ i ∈ s := by simp⟩
  left_inv := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      s : Finset ι
      ⊢ Function.LeftInverse (fun f => ⟨DFinsupp.mk s fun i => ↑(f i), ⋯⟩) fun x =>  …
    -/
    rintro ⟨f, rfl⟩
    /-
      case mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      ⊢ Eq ((fun f_1 => ⟨DFinsupp.mk f.support fun i => ↑(f_1 i), ⋯⟩) ((fun x => DFi …
    -/
    ext i
    /-
      case mk.a.h
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      i : ι
      ⊢ Eq (↑((fun f_1 => ⟨DFinsupp.mk f.support fun i => ↑(f_1 i), ⋯⟩) ((fun x => D …
    -/
    simpa using Eq.symm
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      s : Finset ι
      f : (i : Subtype fun x => Membership.mem s x) → Subtype fun x => Ne x 0
      ⊢ Eq ((fun x => DFinsupp.subtypeSupportEqEquiv.match_2 s (fun x => (i : Subtyp …
    -/
    ext1
    /-
      case h
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      s : Finset ι
      f : (i : Subtype fun x => Membership.mem s x) → Subtype fun x => Ne x 0
      x✝ : Subtype fun x => Membership.mem s x
      ⊢ Eq ((fun x => DFinsupp.subtypeSupportEqEquiv.match_2 s (fun x => (i : Subtyp …
    -/
    simp [Subtype.eta]; rfl
                        /-
                          🎉 no goals
                        -/


/-- Equivalence between all dependent finitely supported functions `f : Π₀ i, β i` and type
of pairs `⟨s : Finset ι, f : ∀ i : s, {x : β i // x ≠ 0}⟩`. -/
@[simps! apply_fst apply_snd_coe]
def sigmaFinsetFunEquiv : (Π₀ i, β i) ≃ Σ s : Finset ι, ∀ i : s, {x : β i // x ≠ 0} :=
  (Equiv.sigmaFiberEquiv DFinsupp.support).symm.trans (.sigmaCongrRight subtypeSupportEqEquiv)


@[simp]
theorem support_zero : (0 : Π₀ i, β i).support = ∅ :=
  rfl


theorem mem_support_iff {f : Π₀ i, β i} {i : ι} : i ∈ f.support ↔ f i ≠ 0 :=
  f.mem_support_toFun _


theorem not_mem_support_iff {f : Π₀ i, β i} {i : ι} : i ∉ f.support ↔ f i = 0 :=
  not_iff_comm.1 mem_support_iff.symm


@[simp]
theorem support_eq_empty {f : Π₀ i, β i} : f.support = ∅ ↔ f = 0 :=
                      /-
                        ι : Type u
                        β : ι → Type v
                        inst✝² : DecidableEq ι
                        inst✝¹ : (i : ι) → Zero (β i)
                        inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
                        f : DFinsupp fun i => β i
                        H : Eq f.support EmptyCollection.emptyCollection
                        ⊢ ∀ (i : ι), Eq (f i) (0 i)
                      -/
                      /-
                        🎉 no goals
                      -/
  ⟨fun H => ext <| by simpa [Finset.ext_iff] using H, by simp +contextual⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


instance decidableZero [∀ (i) (x : β i), Decidable (x = 0)] (f : Π₀ i, β i) : Decidable (f = 0) :=
  f.support'.recOnSubsingleton <| fun s =>
    decidable_of_iff (∀ i ∈ s.val, f i = 0) <| by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝³ : DecidableEq ι
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        inst✝ : (i : ι) → (x : β i) → Decidable (Eq x 0)
        f : DFinsupp fun i => β i
        s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
        ⊢ Iff (∀ (i : ι), Membership.mem (↑s) i → Eq (f i) 0) (Eq f 0)
      -/
      constructor
      /-
        case mp
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝³ : DecidableEq ι
        inst✝² : (i : ι) → Zero (β i)
        inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
        inst✝ : (i : ι) → (x : β i) → Decidable (Eq x 0)
        f : DFinsupp fun i => β i
        s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
        ⊢ (∀ (i : ι), Membership.mem (↑s) i → Eq (f i) 0) → Eq f 0
      -/
      case mpr => rintro rfl _ _; rfl
      case mp =>
        intro hs₁; ext i
        -- This instance prevent consuming `DecidableEq ι` in the next `by_cases`.
        letI := Classical.propDecidable
        by_cases hs₂ : i ∈ s.val
        case pos => exact hs₁ _ hs₂
        case neg => exact (s.prop i).resolve_left hs₂


theorem support_subset_iff {s : Set ι} {f : Π₀ i, β i} : ↑f.support ⊆ s ↔ ∀ i ∉ s, f i = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    s : Set ι
    f : DFinsupp fun i => β i
    ⊢ Iff (HasSubset.Subset (↑f.support) s) (∀ (i : ι), Not (Membership.mem s i) → …
  -/
  simpa [Set.subset_def] using forall_congr' fun i => not_imp_comm
  /-
    🎉 no goals
  -/


theorem support_single_ne_zero {i : ι} {b : β i} (hb : b ≠ 0) : (single i b).support = {i} := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    b : β i
    hb : Ne b 0
    ⊢ Eq (DFinsupp.single i b).support (Singleton.singleton i)
  -/
  ext j; by_cases h : i = j
    /-
      case pos
      ι : Type u
      β : ι → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      i : ι
      b : β i
      hb : Ne b 0
      j : ι
      h : Eq i j
      ⊢ Iff (Membership.mem (DFinsupp.single i b).support j) (Membership.mem (Single …
    -/
  · subst h
    /-
      case pos
      ι : Type u
      β : ι → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      i : ι
      b : β i
      hb : Ne b 0
      ⊢ Iff (Membership.mem (DFinsupp.single i b).support i) (Membership.mem (Single …
    -/
    simp [hb]
    /-
      🎉 no goals
    -/
  /-
    case neg
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    b : β i
    hb : Ne b 0
    j : ι
    h : Not (Eq i j)
    ⊢ Iff (Membership.mem (DFinsupp.single i b).support j) (Membership.mem (Single …
  -/
  simp [Ne.symm h, h]
  /-
    🎉 no goals
  -/


theorem support_single_subset {i : ι} {b : β i} : (single i b).support ⊆ {i} :=
  support_mk'_subset


theorem mapRange_def [∀ (i) (x : β₁ i), Decidable (x ≠ 0)] {f : ∀ i, β₁ i → β₂ i}
    {hf : ∀ i, f i 0 = 0} {g : Π₀ i, β₁ i} :
    mapRange f hf g = mk g.support fun i => f i.1 (g i.1) := by
  /-
    ι : Type u
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β₁ i)
    inst✝¹ : (i : ι) → Zero (β₂ i)
    inst✝ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    g : DFinsupp fun i => β₁ i
    ⊢ Eq (DFinsupp.mapRange f hf g) (DFinsupp.mk g.support fun i => f (↑i) (g ↑i))
  -/
  ext i
  /-
    case h
    ι : Type u
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β₁ i)
    inst✝¹ : (i : ι) → Zero (β₂ i)
    inst✝ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    g : DFinsupp fun i => β₁ i
    i : ι
    ⊢ Eq ((DFinsupp.mapRange f hf g) i) ((DFinsupp.mk g.support fun i => f (↑i) (g …
  -/
                                         /-
                                           🎉 no goals
                                         -/
  by_cases h : g i ≠ 0 <;> simp at h <;> simp [h, hf]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem mapRange_single {f : ∀ i, β₁ i → β₂ i} {hf : ∀ i, f i 0 = 0} {i : ι} {b : β₁ i} :
    mapRange f hf (single i b) = single i (f i b) :=
  DFinsupp.ext fun i' => by
    /-
      ι : Type u
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      i : ι
      b : β₁ i
      i' : ι
      ⊢ Eq ((DFinsupp.mapRange f hf (DFinsupp.single i b)) i') ((DFinsupp.single i ( …
    -/
    by_cases h : i = i'
      /-
        case pos
        ι : Type u
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i
        hf : ∀ (i : ι), Eq (f i 0) 0
        i : ι
        b : β₁ i
        i' : ι
        h : Eq i i'
        ⊢ Eq ((DFinsupp.mapRange f hf (DFinsupp.single i b)) i') ((DFinsupp.single i ( …
      -/
    · subst i'
      /-
        case pos
        ι : Type u
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i
        hf : ∀ (i : ι), Eq (f i 0) 0
        i : ι
        b : β₁ i
        ⊢ Eq ((DFinsupp.mapRange f hf (DFinsupp.single i b)) i) ((DFinsupp.single i (f …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β₁ i)
        inst✝ : (i : ι) → Zero (β₂ i)
        f : (i : ι) → β₁ i → β₂ i
        hf : ∀ (i : ι), Eq (f i 0) 0
        i : ι
        b : β₁ i
        i' : ι
        h : Not (Eq i i')
        ⊢ Eq ((DFinsupp.mapRange f hf (DFinsupp.single i b)) i') ((DFinsupp.single i ( …
      -/
    · simp [h, hf]
      /-
        🎉 no goals
      -/


theorem mapRange_injective (f : ∀ i, β₁ i → β₂ i) (hf : ∀ i, f i 0 = 0) :
    Function.Injective (mapRange f hf) ↔ ∀ i, Function.Injective (f i) :=
  ⟨fun h i x y eq ↦ single_injective (@h (single i x) (single i y) <| by
    /-
      ι : Type u
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β₁ i)
      inst✝ : (i : ι) → Zero (β₂ i)
      f : (i : ι) → β₁ i → β₂ i
      hf : ∀ (i : ι), Eq (f i 0) 0
      h : Function.Injective (DFinsupp.mapRange f hf)
      i : ι
      x y : β₁ i
      eq : Eq (f i x) (f i y)
      ⊢ Eq (DFinsupp.mapRange f hf (DFinsupp.single i x)) (DFinsupp.mapRange f hf (D …
    -/
    simpa using congr_arg _ eq), fun h _ _ eq ↦ DFinsupp.ext fun i ↦ h i congr($eq i)⟩
    /-
      🎉 no goals
    -/


theorem support_mapRange {f : ∀ i, β₁ i → β₂ i} {hf : ∀ i, f i 0 = 0} {g : Π₀ i, β₁ i} :
                                                /-
                                                  ι : Type u
                                                  β₁ : ι → Type v₁
                                                  β₂ : ι → Type v₂
                                                  inst✝⁴ : DecidableEq ι
                                                  inst✝³ : (i : ι) → Zero (β₁ i)
                                                  inst✝² : (i : ι) → Zero (β₂ i)
                                                  inst✝¹ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
                                                  inst✝ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
                                                  f : (i : ι) → β₁ i → β₂ i
                                                  hf : ∀ (i : ι), Eq (f i 0) 0
                                                  g : DFinsupp fun i => β₁ i
                                                  ⊢ HasSubset.Subset (DFinsupp.mapRange f hf g).support g.support
                                                -/
    (mapRange f hf g).support ⊆ g.support := by simp [mapRange_def]
                                                /-
                                                  🎉 no goals
                                                -/


theorem zipWith_def {ι : Type u} {β : ι → Type v} {β₁ : ι → Type v₁} {β₂ : ι → Type v₂}
    [dec : DecidableEq ι] [∀ i : ι, Zero (β i)] [∀ i : ι, Zero (β₁ i)] [∀ i : ι, Zero (β₂ i)]
    [∀ (i : ι) (x : β₁ i), Decidable (x ≠ 0)] [∀ (i : ι) (x : β₂ i), Decidable (x ≠ 0)]
    {f : ∀ i, β₁ i → β₂ i → β i} {hf : ∀ i, f i 0 0 = 0} {g₁ : Π₀ i, β₁ i} {g₂ : Π₀ i, β₂ i} :
    zipWith f hf g₁ g₂ = mk (g₁.support ∪ g₂.support) fun i => f i.1 (g₁ i.1) (g₂ i.1) := by
  /-
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    dec : DecidableEq ι
    inst✝⁴ : (i : ι) → Zero (β i)
    inst✝³ : (i : ι) → Zero (β₁ i)
    inst✝² : (i : ι) → Zero (β₂ i)
    inst✝¹ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    inst✝ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
    f : (i : ι) → β₁ i → β₂ i → β i
    hf : ∀ (i : ι), Eq (f i 0 0) 0
    g₁ : DFinsupp fun i => β₁ i
    g₂ : DFinsupp fun i => β₂ i
    ⊢ Eq (DFinsupp.zipWith f hf g₁ g₂) (DFinsupp.mk (Union.union g₁.support g₂.sup …
  -/
  ext i
  /-
    case h
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    dec : DecidableEq ι
    inst✝⁴ : (i : ι) → Zero (β i)
    inst✝³ : (i : ι) → Zero (β₁ i)
    inst✝² : (i : ι) → Zero (β₂ i)
    inst✝¹ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    inst✝ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
    f : (i : ι) → β₁ i → β₂ i → β i
    hf : ∀ (i : ι), Eq (f i 0 0) 0
    g₁ : DFinsupp fun i => β₁ i
    g₂ : DFinsupp fun i => β₂ i
    i : ι
    ⊢ Eq ((DFinsupp.zipWith f hf g₁ g₂) i) ((DFinsupp.mk (Union.union g₁.support g …
  -/
  by_cases h1 : g₁ i ≠ 0 <;> by_cases h2 : g₂ i ≠ 0 <;> simp only [not_not, Ne] at h1 h2 <;>
    /-
      case pos
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      dec : DecidableEq ι
      inst✝⁴ : (i : ι) → Zero (β i)
      inst✝³ : (i : ι) → Zero (β₁ i)
      inst✝² : (i : ι) → Zero (β₂ i)
      inst✝¹ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
      inst✝ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
      f : (i : ι) → β₁ i → β₂ i → β i
      hf : ∀ (i : ι), Eq (f i 0 0) 0
      g₁ : DFinsupp fun i => β₁ i
      g₂ : DFinsupp fun i => β₂ i
      i : ι
      h1 : Not (Eq (g₁ i) 0)
      h2 : Not (Eq (g₂ i) 0)
      ⊢ Eq ((DFinsupp.zipWith f hf g₁ g₂) i) ((DFinsupp.mk (Union.union g₁.support g …
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
    simp [h1, h2, hf]
    /-
      🎉 no goals
    -/


theorem support_zipWith {f : ∀ i, β₁ i → β₂ i → β i} {hf : ∀ i, f i 0 0 = 0} {g₁ : Π₀ i, β₁ i}
    {g₂ : Π₀ i, β₂ i} : (zipWith f hf g₁ g₂).support ⊆ g₁.support ∪ g₂.support := by
  /-
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : (i : ι) → Zero (β i)
    inst✝⁴ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    inst✝³ : (i : ι) → Zero (β₁ i)
    inst✝² : (i : ι) → Zero (β₂ i)
    inst✝¹ : (i : ι) → (x : β₁ i) → Decidable (Ne x 0)
    inst✝ : (i : ι) → (x : β₂ i) → Decidable (Ne x 0)
    f : (i : ι) → β₁ i → β₂ i → β i
    hf : ∀ (i : ι), Eq (f i 0 0) 0
    g₁ : DFinsupp fun i => β₁ i
    g₂ : DFinsupp fun i => β₂ i
    ⊢ HasSubset.Subset (DFinsupp.zipWith f hf g₁ g₂).support (Union.union g₁.suppo …
  -/
  simp [zipWith_def]
  /-
    🎉 no goals
  -/


theorem erase_def (i : ι) (f : Π₀ i, β i) : f.erase i = mk (f.support.erase i) fun j => f j.1 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.erase i f) (DFinsupp.mk (f.support.erase i) fun j => f ↑j)
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : DFinsupp fun i => β i
    j : ι
    ⊢ Eq ((DFinsupp.erase i f) j) ((DFinsupp.mk (f.support.erase i) fun j => f ↑j) …
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
  by_cases h1 : j = i <;> by_cases h2 : f j ≠ 0 <;> simp at h2 <;> simp [h1, h2]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem support_erase (i : ι) (f : Π₀ i, β i) : (f.erase i).support = f.support.erase i := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.erase i f).support (f.support.erase i)
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : DFinsupp fun i => β i
    j : ι
    ⊢ Iff (Membership.mem (DFinsupp.erase i f).support j) (Membership.mem (f.suppo …
  -/
  by_cases h1 : j = i
  · simp only [h1, mem_support_toFun, erase_apply, ite_true, ne_eq, not_true, not_not,
      Finset.mem_erase, false_and]
  /-
    case neg
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    i : ι
    f : DFinsupp fun i => β i
    j : ι
    h1 : Not (Eq j i)
    ⊢ Iff (Membership.mem (DFinsupp.erase i f).support j) (Membership.mem (f.suppo …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  by_cases h2 : f j ≠ 0 <;> simp at h2 <;> simp [h1, h2]
                                           /-
                                             🎉 no goals
                                           -/


theorem support_update_ne_zero (f : Π₀ i, β i) (i : ι) {b : β i} (h : b ≠ 0) :
    support (f.update i b) = insert i f.support := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    h : Ne b 0
    ⊢ Eq (f.update i b).support (Insert.insert i f.support)
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → Zero (β i)
    inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    h : Ne b 0
    j : ι
    ⊢ Iff (Membership.mem (f.update i b).support j) (Membership.mem (Insert.insert …
  -/
  rcases eq_or_ne i j with (rfl | hi)
    /-
      case h.inl
      ι : Type u
      β : ι → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      h : Ne b 0
      ⊢ Iff (Membership.mem (f.update i b).support i) (Membership.mem (Insert.insert …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      β : ι → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → Zero (β i)
      inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      h : Ne b 0
      j : ι
      hi : Ne i j
      ⊢ Iff (Membership.mem (f.update i b).support j) (Membership.mem (Insert.insert …
    -/
  · simp [hi.symm]
    /-
      🎉 no goals
    -/


theorem support_update (f : Π₀ i, β i) (i : ι) (b : β i) [Decidable (b = 0)] :
    support (f.update i b) = if b = 0 then support (f.erase i) else insert i f.support := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    inst✝ : Decidable (Eq b 0)
    ⊢ Eq (f.update i b).support (ite (Eq b 0) (DFinsupp.erase i f).support (Insert …
  -/
  ext j
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    f : DFinsupp fun i => β i
    i : ι
    b : β i
    inst✝ : Decidable (Eq b 0)
    j : ι
    ⊢ Iff (Membership.mem (f.update i b).support j) (Membership.mem (ite (Eq b 0)  …
  -/
  split_ifs with hb
    /-
      case pos
      ι : Type u
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      inst✝ : Decidable (Eq b 0)
      j : ι
      hb : Eq b 0
      ⊢ Iff (Membership.mem (f.update i b).support j) (Membership.mem (DFinsupp.eras …
    -/
  · subst hb
    /-
      case pos
      ι : Type u
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      i j : ι
      inst✝ : Decidable (Eq 0 0)
      ⊢ Iff (Membership.mem (f.update i 0).support j) (Membership.mem (DFinsupp.eras …
    -/
    simp [update_eq_erase, support_erase]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u
      β : ι → Type v
      inst✝³ : DecidableEq ι
      inst✝² : (i : ι) → Zero (β i)
      inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
      f : DFinsupp fun i => β i
      i : ι
      b : β i
      inst✝ : Decidable (Eq b 0)
      j : ι
      hb : Not (Eq b 0)
      ⊢ Iff (Membership.mem (f.update i b).support j) (Membership.mem (Insert.insert …
    -/
  · rw [support_update_ne_zero f _ hb]
    /-
      🎉 no goals
    -/


theorem filter_def (f : Π₀ i, β i) : f.filter p = mk (f.support.filter p) fun i => f i.1 := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    p : ι → Prop
    inst✝ : DecidablePred p
    f : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.filter p f) (DFinsupp.mk (Finset.filter p f.support) fun i => f …
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
  ext i; by_cases h1 : p i <;> by_cases h2 : f i ≠ 0 <;> simp at h2 <;> simp [h1, h2]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem support_filter (f : Π₀ i, β i) : (f.filter p).support = {x ∈ f.support | p x} := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    p : ι → Prop
    inst✝ : DecidablePred p
    f : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.filter p f).support (Finset.filter (fun x => p x) f.support)
  -/
                              /-
                                🎉 no goals
                              -/
  ext i; by_cases h : p i <;> simp [h]
                              /-
                                🎉 no goals
                              -/


theorem subtypeDomain_def (f : Π₀ i, β i) :
    f.subtypeDomain p = mk (f.support.subtype p) fun i => f i := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    p : ι → Prop
    inst✝ : DecidablePred p
    f : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.subtypeDomain p f) (DFinsupp.mk (Finset.subtype p f.support) fu …
  -/
                                   /-
                                     🎉 no goals
                                   -/
  ext i; by_cases h2 : f i ≠ 0 <;> try simp at h2; dsimp; simp [h2]
                                   /-
                                     🎉 no goals
                                   -/


@[simp, nolint simpNF] -- Porting note: simpNF claims that LHS does not simplify, but it does
theorem support_subtypeDomain {f : Π₀ i, β i} :
    (subtypeDomain p f).support = f.support.subtype p := by
  /-
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    p : ι → Prop
    inst✝ : DecidablePred p
    f : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.subtypeDomain p f).support (Finset.subtype p f.support)
  -/
  ext i
  /-
    case h
    ι : Type u
    β : ι → Type v
    inst✝³ : DecidableEq ι
    inst✝² : (i : ι) → Zero (β i)
    inst✝¹ : (i : ι) → (x : β i) → Decidable (Ne x 0)
    p : ι → Prop
    inst✝ : DecidablePred p
    f : DFinsupp fun i => β i
    i : Subtype p
    ⊢ Iff (Membership.mem (DFinsupp.subtypeDomain p f).support i) (Membership.mem  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem support_add [∀ i, AddZeroClass (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)]
    {g₁ g₂ : Π₀ i, β i} : (g₁ + g₂).support ⊆ g₁.support ∪ g₂.support :=
  support_zipWith


@[simp]
theorem support_neg [∀ i, AddGroup (β i)] [∀ (i) (x : β i), Decidable (x ≠ 0)] {f : Π₀ i, β i} :
                                   /-
                                     ι : Type u
                                     β : ι → Type v
                                     inst✝² : DecidableEq ι
                                     inst✝¹ : (i : ι) → AddGroup (β i)
                                     inst✝ : (i : ι) → (x : β i) → Decidable (Ne x 0)
                                     f : DFinsupp fun i => β i
                                     ⊢ Eq (Neg.neg f).support f.support
                                   -/
    support (-f) = support f := by ext i; simp
                                          /-
                                            🎉 no goals
                                          -/


instance [∀ i, Zero (β i)] [∀ i, DecidableEq (β i)] : DecidableEq (Π₀ i, β i) := fun f g =>
  decidable_of_iff (f.support = g.support ∧ ∀ i ∈ f.support, f i = g i)
    ⟨fun ⟨h₁, h₂⟩ => ext fun i => if h : i ∈ f.support then h₂ i h else by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → DecidableEq (β i)
        f g : DFinsupp fun i => β i
        x✝ : And (Eq f.support g.support) (∀ (i : ι), Membership.mem f.support i → Eq  …
        h₁ : Eq f.support g.support
        h₂ : ∀ (i : ι), Membership.mem f.support i → Eq (f i) (g i)
        i : ι
        h : Not (Membership.mem f.support i)
        ⊢ Eq (f i) (g i)
      -/
      have hf : f i = 0 := by rwa [mem_support_iff, not_not] at h
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → DecidableEq (β i)
        f g : DFinsupp fun i => β i
        x✝ : And (Eq f.support g.support) (∀ (i : ι), Membership.mem f.support i → Eq  …
        h₁ : Eq f.support g.support
        h₂ : ∀ (i : ι), Membership.mem f.support i → Eq (f i) (g i)
        i : ι
        h : Not (Membership.mem f.support i)
        hf : Eq (f i) 0
        ⊢ Eq (f i) (g i)
      -/
      have hg : g i = 0 := by rwa [h₁, mem_support_iff, not_not] at h
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → Zero (β i)
        inst✝ : (i : ι) → DecidableEq (β i)
        f g : DFinsupp fun i => β i
        x✝ : And (Eq f.support g.support) (∀ (i : ι), Membership.mem f.support i → Eq  …
        h₁ : Eq f.support g.support
        h₂ : ∀ (i : ι), Membership.mem f.support i → Eq (f i) (g i)
        i : ι
        h : Not (Membership.mem f.support i)
        hf : Eq (f i) 0
        hg : Eq (g i) 0
        ⊢ Eq (f i) (g i)
      -/
      rw [hf, hg],
      /-
        🎉 no goals
      -/
        /-
          ι : Type u
          γ : Type w
          β : ι → Type v
          β₁ : ι → Type v₁
          β₂ : ι → Type v₂
          inst✝² : DecidableEq ι
          inst✝¹ : (i : ι) → Zero (β i)
          inst✝ : (i : ι) → DecidableEq (β i)
          f g : DFinsupp fun i => β i
          ⊢ Eq f g → And (Eq f.support g.support) (∀ (i : ι), Membership.mem f.support i …
        -/
     by rintro rfl; simp⟩
                    /-
                      🎉 no goals
                    -/


/-- Reindexing (and possibly removing) terms of a dfinsupp. -/
noncomputable def comapDomain [∀ i, Zero (β i)] (h : κ → ι) (hh : Function.Injective h)
    (f : Π₀ i, β i) : Π₀ k, β (h k) where
  toFun x := f (h x)
  support' :=
    f.support'.map fun s =>
      ⟨(s.1.finite_toSet.preimage hh.injOn).toFinset.val, fun x =>
        (s.prop (h x)).imp_left fun hx => (Set.Finite.mem_toFinset _).mpr <| hx⟩


@[simp]
theorem comapDomain_apply [∀ i, Zero (β i)] (h : κ → ι) (hh : Function.Injective h) (f : Π₀ i, β i)
    (k : κ) : comapDomain h hh f k = f (h k) :=
  rfl


@[simp]
theorem comapDomain_zero [∀ i, Zero (β i)] (h : κ → ι) (hh : Function.Injective h) :
    comapDomain h hh (0 : Π₀ i, β i) = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    hh : Function.Injective h
    ⊢ Eq (DFinsupp.comapDomain h hh 0) 0
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    hh : Function.Injective h
    i✝ : κ
    ⊢ Eq ((DFinsupp.comapDomain h hh 0) i✝) (0 i✝)
  -/
  rw [zero_apply, comapDomain_apply, zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comapDomain_add [∀ i, AddZeroClass (β i)] (h : κ → ι) (hh : Function.Injective h)
    (f g : Π₀ i, β i) : comapDomain h hh (f + g) = comapDomain h hh f + comapDomain h hh g := by
  /-
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    h : κ → ι
    hh : Function.Injective h
    f g : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.comapDomain h hh (HAdd.hAdd f g)) (HAdd.hAdd (DFinsupp.comapDom …
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    h : κ → ι
    hh : Function.Injective h
    f g : DFinsupp fun i => β i
    i✝ : κ
    ⊢ Eq ((DFinsupp.comapDomain h hh (HAdd.hAdd f g)) i✝) ((HAdd.hAdd (DFinsupp.co …
  -/
  rw [add_apply, comapDomain_apply, comapDomain_apply, comapDomain_apply, add_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comapDomain_single [DecidableEq ι] [DecidableEq κ] [∀ i, Zero (β i)] (h : κ → ι)
    (hh : Function.Injective h) (k : κ) (x : β (h k)) :
    comapDomain h hh (single (h k) x) = single k x := by
  /-
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableEq κ
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    hh : Function.Injective h
    k : κ
    x : β (h k)
    ⊢ Eq (DFinsupp.comapDomain h hh (DFinsupp.single (h k) x)) (DFinsupp.single k x)
  -/
  ext i
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableEq κ
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    hh : Function.Injective h
    k : κ
    x : β (h k)
    i : κ
    ⊢ Eq ((DFinsupp.comapDomain h hh (DFinsupp.single (h k) x)) i) ((DFinsupp.sing …
  -/
  rw [comapDomain_apply]
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableEq κ
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    hh : Function.Injective h
    k : κ
    x : β (h k)
    i : κ
    ⊢ Eq ((DFinsupp.single (h k) x) (h i)) ((DFinsupp.single k x) i)
  -/
  obtain rfl | hik := Decidable.eq_or_ne i k
    /-
      case h.inl
      ι : Type u
      β : ι → Type v
      κ : Type u_1
      inst✝² : DecidableEq ι
      inst✝¹ : DecidableEq κ
      inst✝ : (i : ι) → Zero (β i)
      h : κ → ι
      hh : Function.Injective h
      i : κ
      x : β (h i)
      ⊢ Eq ((DFinsupp.single (h i) x) (h i)) ((DFinsupp.single i x) i)
    -/
  · rw [single_eq_same, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      β : ι → Type v
      κ : Type u_1
      inst✝² : DecidableEq ι
      inst✝¹ : DecidableEq κ
      inst✝ : (i : ι) → Zero (β i)
      h : κ → ι
      hh : Function.Injective h
      k : κ
      x : β (h k)
      i : κ
      hik : Ne i k
      ⊢ Eq ((DFinsupp.single (h k) x) (h i)) ((DFinsupp.single k x) i)
    -/
  · rw [single_eq_of_ne hik.symm, single_eq_of_ne (hh.ne hik.symm)]
    /-
      🎉 no goals
    -/


/-- A computable version of comap_domain when an explicit left inverse is provided. -/
def comapDomain' [∀ i, Zero (β i)] (h : κ → ι) {h' : ι → κ} (hh' : Function.LeftInverse h' h)
    (f : Π₀ i, β i) : Π₀ k, β (h k) where
  toFun x := f (h x)
  support' :=
    f.support'.map fun s =>
      ⟨Multiset.map h' s.1, fun x =>
        (s.prop (h x)).imp_left fun hx => Multiset.mem_map.mpr ⟨_, hx, hh' _⟩⟩


@[simp]
theorem comapDomain'_apply [∀ i, Zero (β i)] (h : κ → ι) {h' : ι → κ}
    (hh' : Function.LeftInverse h' h) (f : Π₀ i, β i) (k : κ) : comapDomain' h hh' f k = f (h k) :=
  rfl


@[simp]
theorem comapDomain'_zero [∀ i, Zero (β i)] (h : κ → ι) {h' : ι → κ}
    (hh' : Function.LeftInverse h' h) : comapDomain' h hh' (0 : Π₀ i, β i) = 0 := by
  /-
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    ⊢ Eq (DFinsupp.comapDomain' h hh' 0) 0
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    i✝ : κ
    ⊢ Eq ((DFinsupp.comapDomain' h hh' 0) i✝) (0 i✝)
  -/
  rw [zero_apply, comapDomain'_apply, zero_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comapDomain'_add [∀ i, AddZeroClass (β i)] (h : κ → ι) {h' : ι → κ}
    (hh' : Function.LeftInverse h' h) (f g : Π₀ i, β i) :
    comapDomain' h hh' (f + g) = comapDomain' h hh' f + comapDomain' h hh' g := by
  /-
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    f g : DFinsupp fun i => β i
    ⊢ Eq (DFinsupp.comapDomain' h hh' (HAdd.hAdd f g)) (HAdd.hAdd (DFinsupp.comapD …
  -/
  ext
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝ : (i : ι) → AddZeroClass (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    f g : DFinsupp fun i => β i
    i✝ : κ
    ⊢ Eq ((DFinsupp.comapDomain' h hh' (HAdd.hAdd f g)) i✝) ((HAdd.hAdd (DFinsupp. …
  -/
  rw [add_apply, comapDomain'_apply, comapDomain'_apply, comapDomain'_apply, add_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comapDomain'_single [DecidableEq ι] [DecidableEq κ] [∀ i, Zero (β i)] (h : κ → ι)
    {h' : ι → κ} (hh' : Function.LeftInverse h' h) (k : κ) (x : β (h k)) :
    comapDomain' h hh' (single (h k) x) = single k x := by
  /-
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableEq κ
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    k : κ
    x : β (h k)
    ⊢ Eq (DFinsupp.comapDomain' h hh' (DFinsupp.single (h k) x)) (DFinsupp.single  …
  -/
  ext i
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableEq κ
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    k : κ
    x : β (h k)
    i : κ
    ⊢ Eq ((DFinsupp.comapDomain' h hh' (DFinsupp.single (h k) x)) i) ((DFinsupp.si …
  -/
  rw [comapDomain'_apply]
  /-
    case h
    ι : Type u
    β : ι → Type v
    κ : Type u_1
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableEq κ
    inst✝ : (i : ι) → Zero (β i)
    h : κ → ι
    h' : ι → κ
    hh' : Function.LeftInverse h' h
    k : κ
    x : β (h k)
    i : κ
    ⊢ Eq ((DFinsupp.single (h k) x) (h i)) ((DFinsupp.single k x) i)
  -/
  obtain rfl | hik := Decidable.eq_or_ne i k
    /-
      case h.inl
      ι : Type u
      β : ι → Type v
      κ : Type u_1
      inst✝² : DecidableEq ι
      inst✝¹ : DecidableEq κ
      inst✝ : (i : ι) → Zero (β i)
      h : κ → ι
      h' : ι → κ
      hh' : Function.LeftInverse h' h
      i : κ
      x : β (h i)
      ⊢ Eq ((DFinsupp.single (h i) x) (h i)) ((DFinsupp.single i x) i)
    -/
  · rw [single_eq_same, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      ι : Type u
      β : ι → Type v
      κ : Type u_1
      inst✝² : DecidableEq ι
      inst✝¹ : DecidableEq κ
      inst✝ : (i : ι) → Zero (β i)
      h : κ → ι
      h' : ι → κ
      hh' : Function.LeftInverse h' h
      k : κ
      x : β (h k)
      i : κ
      hik : Ne i k
      ⊢ Eq ((DFinsupp.single (h k) x) (h i)) ((DFinsupp.single k x) i)
    -/
  · rw [single_eq_of_ne hik.symm, single_eq_of_ne (hh'.injective.ne hik.symm)]
    /-
      🎉 no goals
    -/


/-- Reindexing terms of a dfinsupp.

This is the dfinsupp version of `Equiv.piCongrLeft'`. -/
@[simps apply]
def equivCongrLeft [∀ i, Zero (β i)] (h : ι ≃ κ) : (Π₀ i, β i) ≃ Π₀ k, β (h.symm k) where
  toFun := comapDomain' h.symm h.right_inv
  invFun f :=
    mapRange (fun i => Equiv.cast <| congr_arg β <| h.symm_apply_apply i)
                                                    /-
                                                      ι : Type u
                                                      γ : Type w
                                                      β : ι → Type v
                                                      β₁ : ι → Type v₁
                                                      β₂ : ι → Type v₂
                                                      κ : Type u_1
                                                      inst✝ : (i : ι) → Zero (β i)
                                                      h : Equiv ι κ
                                                      f : DFinsupp fun k => β (h.symm k)
                                                      i : ι
                                                      ⊢ HEq 0 0
                                                    -/
      (fun i => (Equiv.cast_eq_iff_heq _).mpr <| by rw [Equiv.symm_apply_apply])
                                                    /-
                                                      🎉 no goals
                                                    -/
      (@comapDomain' _ _ _ _ h _ h.left_inv f)
  left_inv f := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      inst✝ : (i : ι) → Zero (β i)
      h : Equiv ι κ
      f : DFinsupp fun i => β i
      ⊢ Eq ((fun f => DFinsupp.mapRange (fun i => ⇑(Equiv.cast ⋯)) ⋯ (DFinsupp.comap …
    -/
    ext i
    rw [mapRange_apply, comapDomain'_apply, comapDomain'_apply, Equiv.cast_eq_iff_heq,
      h.symm_apply_apply]
  right_inv f := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      inst✝ : (i : ι) → Zero (β i)
      h : Equiv ι κ
      f : DFinsupp fun k => β (h.symm k)
      ⊢ Eq (DFinsupp.comapDomain' ⇑h.symm ⋯ ((fun f => DFinsupp.mapRange (fun i => ⇑ …
    -/
    ext k
    rw [comapDomain'_apply, mapRange_apply, comapDomain'_apply, Equiv.cast_eq_iff_heq,
      h.apply_symm_apply]


instance hasAdd₂ [∀ i j, AddZeroClass (δ i j)] : Add (Π₀ (i : ι) (j : α i), δ i j) :=
  inferInstance
  -- @DFinsupp.hasAdd ι (fun i => Π₀ j, δ i j) _


instance addZeroClass₂ [∀ i j, AddZeroClass (δ i j)] : AddZeroClass (Π₀ (i : ι) (j : α i), δ i j) :=
  inferInstance
  -- @DFinsupp.addZeroClass ι (fun i => Π₀ j, δ i j) _


instance addMonoid₂ [∀ i j, AddMonoid (δ i j)] : AddMonoid (Π₀ (i : ι) (j : α i), δ i j) :=
  inferInstance
  -- @DFinsupp.addMonoid ι (fun i => Π₀ j, δ i j) _


/-- Adds a term to a dfinsupp, making a dfinsupp indexed by an `Option`.

This is the dfinsupp version of `Option.rec`. -/
def extendWith [∀ i, Zero (α i)] (a : α none) (f : Π₀ i, α (some i)) : Π₀ i, α i where
  toFun := fun i ↦ match i with | none => a | some _ => f _
  support' :=
    f.support'.map fun s =>
      ⟨none ::ₘ Multiset.map some s.1, fun i =>
        Option.rec (Or.inl <| Multiset.mem_cons_self _ _)
          (fun i =>
            (s.prop i).imp_left fun h => Multiset.mem_cons_of_mem <| Multiset.mem_map_of_mem _ h)
          i⟩


@[simp]
theorem extendWith_none [∀ i, Zero (α i)] (f : Π₀ i, α (some i)) (a : α none) :
    f.extendWith a none = a :=
  rfl


@[simp]
theorem extendWith_some [∀ i, Zero (α i)] (f : Π₀ i, α (some i)) (a : α none) (i : ι) :
    f.extendWith a (some i) = f i :=
  rfl


@[simp]
theorem extendWith_single_zero [DecidableEq ι] [∀ i, Zero (α i)] (i : ι) (x : α (some i)) :
    (single i x).extendWith 0 = single (some i) x := by
  /-
    ι : Type u
    α : Option ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : Option ι) → Zero (α i)
    i : ι
    x : α (Option.some i)
    ⊢ Eq (DFinsupp.extendWith 0 (DFinsupp.single i x)) (DFinsupp.single (Option.so …
  -/
  ext (_ | j)
    /-
      case h.none
      ι : Type u
      α : Option ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : Option ι) → Zero (α i)
      i : ι
      x : α (Option.some i)
      ⊢ Eq ((DFinsupp.extendWith 0 (DFinsupp.single i x)) Option.none) ((DFinsupp.si …
    -/
  · rw [extendWith_none, single_eq_of_ne (Option.some_ne_none _)]
    /-
      🎉 no goals
    -/
    /-
      case h.some
      ι : Type u
      α : Option ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : Option ι) → Zero (α i)
      i : ι
      x : α (Option.some i)
      j : ι
      ⊢ Eq ((DFinsupp.extendWith 0 (DFinsupp.single i x)) (Option.some j)) ((DFinsup …
    -/
  · rw [extendWith_some]
    /-
      case h.some
      ι : Type u
      α : Option ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : Option ι) → Zero (α i)
      i : ι
      x : α (Option.some i)
      j : ι
      ⊢ Eq ((DFinsupp.single i x) j) ((DFinsupp.single (Option.some i) x) (Option.so …
    -/
    obtain rfl | hij := Decidable.eq_or_ne i j
      /-
        case h.some.inl
        ι : Type u
        α : Option ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : Option ι) → Zero (α i)
        i : ι
        x : α (Option.some i)
        ⊢ Eq ((DFinsupp.single i x) i) ((DFinsupp.single (Option.some i) x) (Option.so …
      -/
    · rw [single_eq_same, single_eq_same]
      /-
        🎉 no goals
      -/
      /-
        case h.some.inr
        ι : Type u
        α : Option ι → Type v
        inst✝¹ : DecidableEq ι
        inst✝ : (i : Option ι) → Zero (α i)
        i : ι
        x : α (Option.some i)
        j : ι
        hij : Ne i j
        ⊢ Eq ((DFinsupp.single i x) j) ((DFinsupp.single (Option.some i) x) (Option.so …
      -/
    · rw [single_eq_of_ne hij, single_eq_of_ne ((Option.some_injective _).ne hij)]
      /-
        🎉 no goals
      -/


@[simp]
theorem extendWith_zero [DecidableEq ι] [∀ i, Zero (α i)] (x : α none) :
    (0 : Π₀ i, α (some i)).extendWith x = single none x := by
  /-
    ι : Type u
    α : Option ι → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : Option ι) → Zero (α i)
    x : α Option.none
    ⊢ Eq (DFinsupp.extendWith x 0) (DFinsupp.single Option.none x)
  -/
  ext (_ | j)
    /-
      case h.none
      ι : Type u
      α : Option ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : Option ι) → Zero (α i)
      x : α Option.none
      ⊢ Eq ((DFinsupp.extendWith x 0) Option.none) ((DFinsupp.single Option.none x)  …
    -/
  · rw [extendWith_none, single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h.some
      ι : Type u
      α : Option ι → Type v
      inst✝¹ : DecidableEq ι
      inst✝ : (i : Option ι) → Zero (α i)
      x : α Option.none
      j : ι
      ⊢ Eq ((DFinsupp.extendWith x 0) (Option.some j)) ((DFinsupp.single Option.none …
    -/
  · rw [extendWith_some, single_eq_of_ne (Option.some_ne_none _).symm, zero_apply]
    /-
      🎉 no goals
    -/


/-- Bijection obtained by separating the term of index `none` of a dfinsupp over `Option ι`.

This is the dfinsupp version of `Equiv.piOptionEquivProd`. -/
@[simps]
noncomputable def equivProdDFinsupp [∀ i, Zero (α i)] :
    (Π₀ i, α i) ≃ α none × Π₀ i, α (some i) where
  toFun f := (f none, comapDomain some (Option.some_injective _) f)
  invFun f := f.2.extendWith f.1
  left_inv f := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : Option ι → Type v
      inst✝ : (i : Option ι) → Zero (α i)
      f : DFinsupp fun i => α i
      ⊢ Eq ((fun f => DFinsupp.extendWith f.1 f.2) ((fun f => { fst := f Option.none …
    -/
    ext i; cases' i with i
      /-
        case h.none
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        κ : Type u_1
        α : Option ι → Type v
        inst✝ : (i : Option ι) → Zero (α i)
        f : DFinsupp fun i => α i
        ⊢ Eq (((fun f => DFinsupp.extendWith f.1 f.2) ((fun f => { fst := f Option.non …
      -/
    · rw [extendWith_none]
      /-
        🎉 no goals
      -/
      /-
        case h.some
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        κ : Type u_1
        α : Option ι → Type v
        inst✝ : (i : Option ι) → Zero (α i)
        f : DFinsupp fun i => α i
        i : ι
        ⊢ Eq (((fun f => DFinsupp.extendWith f.1 f.2) ((fun f => { fst := f Option.non …
      -/
    · rw [extendWith_some, comapDomain_apply]
      /-
        🎉 no goals
      -/
  right_inv x := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : Option ι → Type v
      inst✝ : (i : Option ι) → Zero (α i)
      x : Prod (α Option.none) (DFinsupp fun i => α (Option.some i))
      ⊢ Eq ((fun f => { fst := f Option.none, snd := DFinsupp.comapDomain Option.som …
    -/
    dsimp only
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : Option ι → Type v
      inst✝ : (i : Option ι) → Zero (α i)
      x : Prod (α Option.none) (DFinsupp fun i => α (Option.some i))
      ⊢ Eq { fst := (DFinsupp.extendWith x.1 x.2) Option.none, snd := DFinsupp.comap …
    -/
    ext
      /-
        case fst
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        κ : Type u_1
        α : Option ι → Type v
        inst✝ : (i : Option ι) → Zero (α i)
        x : Prod (α Option.none) (DFinsupp fun i => α (Option.some i))
        ⊢ Eq { fst := (DFinsupp.extendWith x.1 x.2) Option.none, snd := DFinsupp.comap …
      -/
    · exact extendWith_none x.snd _
      /-
        🎉 no goals
      -/
      /-
        case snd.h
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        κ : Type u_1
        α : Option ι → Type v
        inst✝ : (i : Option ι) → Zero (α i)
        x : Prod (α Option.none) (DFinsupp fun i => α (Option.some i))
        i✝ : ι
        ⊢ Eq ({ fst := (DFinsupp.extendWith x.1 x.2) Option.none, snd := DFinsupp.coma …
      -/
    · rw [comapDomain_apply, extendWith_some]
      /-
        🎉 no goals
      -/


theorem equivProdDFinsupp_add [∀ i, AddZeroClass (α i)] (f g : Π₀ i, α i) :
    equivProdDFinsupp (f + g) = equivProdDFinsupp f + equivProdDFinsupp g :=
  Prod.ext (add_apply _ _ _) (comapDomain_add _ (Option.some_injective _) _ _)


theorem mapRange_add (f : ∀ i, β₁ i → β₂ i) (hf : ∀ i, f i 0 = 0)
    (hf' : ∀ i x y, f i (x + y) = f i x + f i y) (g₁ g₂ : Π₀ i, β₁ i) :
    mapRange f hf (g₁ + g₂) = mapRange f hf g₁ + mapRange f hf g₂ := by
  /-
    ι : Type u
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
    inst✝ : (i : ι) → AddZeroClass (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    hf' : ∀ (i : ι) (x y : β₁ i), Eq (f i (HAdd.hAdd x y)) (HAdd.hAdd (f i x) (f i …
    g₁ g₂ : DFinsupp fun i => β₁ i
    ⊢ Eq (DFinsupp.mapRange f hf (HAdd.hAdd g₁ g₂)) (HAdd.hAdd (DFinsupp.mapRange  …
  -/
  ext
  /-
    case h
    ι : Type u
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
    inst✝ : (i : ι) → AddZeroClass (β₂ i)
    f : (i : ι) → β₁ i → β₂ i
    hf : ∀ (i : ι), Eq (f i 0) 0
    hf' : ∀ (i : ι) (x y : β₁ i), Eq (f i (HAdd.hAdd x y)) (HAdd.hAdd (f i x) (f i …
    g₁ g₂ : DFinsupp fun i => β₁ i
    i✝ : ι
    ⊢ Eq ((DFinsupp.mapRange f hf (HAdd.hAdd g₁ g₂)) i✝) ((HAdd.hAdd (DFinsupp.map …
  -/
  simp only [mapRange_apply f, coe_add, Pi.add_apply, hf']
  /-
    🎉 no goals
  -/


/-- `DFinsupp.mapRange` as an `AddMonoidHom`. -/
@[simps apply]
def mapRange.addMonoidHom (f : ∀ i, β₁ i →+ β₂ i) : (Π₀ i, β₁ i) →+ Π₀ i, β₂ i where
  toFun := mapRange (fun i x => f i x) fun i => (f i).map_zero
  map_zero' := mapRange_zero _ _
  map_add' := mapRange_add _ (fun i => (f i).map_zero) fun i => (f i).map_add


@[simp]
theorem mapRange.addMonoidHom_id :
    (mapRange.addMonoidHom fun i => AddMonoidHom.id (β₂ i)) = AddMonoidHom.id _ :=
  AddMonoidHom.ext mapRange_id


theorem mapRange.addMonoidHom_comp (f : ∀ i, β₁ i →+ β₂ i) (f₂ : ∀ i, β i →+ β₁ i) :
    (mapRange.addMonoidHom fun i => (f i).comp (f₂ i)) =
      (mapRange.addMonoidHom f).comp (mapRange.addMonoidHom f₂) := by
  /-
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
    inst✝ : (i : ι) → AddZeroClass (β₂ i)
    f : (i : ι) → AddMonoidHom (β₁ i) (β₂ i)
    f₂ : (i : ι) → AddMonoidHom (β i) (β₁ i)
    ⊢ Eq (DFinsupp.mapRange.addMonoidHom fun i => (f i).comp (f₂ i)) ((DFinsupp.ma …
  -/
  refine AddMonoidHom.ext <| mapRange_comp (fun i x => f i x) (fun i x => f₂ i x) ?_ ?_ ?_
    /-
      case refine_1
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
      inst✝ : (i : ι) → AddZeroClass (β₂ i)
      f : (i : ι) → AddMonoidHom (β₁ i) (β₂ i)
      f₂ : (i : ι) → AddMonoidHom (β i) (β₁ i)
      ⊢ ∀ (i : ι), Eq ((fun i x => (f i) x) i 0) 0
    -/
  · intros; apply map_zero
            /-
              🎉 no goals
            -/
    /-
      case refine_2
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
      inst✝ : (i : ι) → AddZeroClass (β₂ i)
      f : (i : ι) → AddMonoidHom (β₁ i) (β₂ i)
      f₂ : (i : ι) → AddMonoidHom (β i) (β₁ i)
      ⊢ ∀ (i : ι), Eq ((fun i x => (f₂ i) x) i 0) 0
    -/
  · intros; apply map_zero
            /-
              🎉 no goals
            -/
    /-
      case refine_3
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
      inst✝ : (i : ι) → AddZeroClass (β₂ i)
      f : (i : ι) → AddMonoidHom (β₁ i) (β₂ i)
      f₂ : (i : ι) → AddMonoidHom (β i) (β₁ i)
      ⊢ ∀ (i : ι), Eq (Function.comp ((fun i x => (f i) x) i) ((fun i x => (f₂ i) x) …
    -/
  · intros; dsimp; simp only [map_zero]
                   /-
                     🎉 no goals
                   -/


/-- `DFinsupp.mapRange.addMonoidHom` as an `AddEquiv`. -/
@[simps apply]
def mapRange.addEquiv (e : ∀ i, β₁ i ≃+ β₂ i) : (Π₀ i, β₁ i) ≃+ Π₀ i, β₂ i :=
  { mapRange.addMonoidHom fun i =>
      (e i).toAddMonoidHom with
    toFun := mapRange (fun i x => e i x) fun i => (e i).map_zero
    invFun := mapRange (fun i x => (e i).symm x) fun i => (e i).symm.map_zero
    left_inv := fun x => by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → AddZeroClass (β i)
        inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
        inst✝ : (i : ι) → AddZeroClass (β₂ i)
        e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
        x : DFinsupp fun i => β₁ i
        ⊢ Eq (DFinsupp.mapRange (fun i x => (e i).symm x) ⋯ (DFinsupp.mapRange (fun i  …
      -/
      rw [← mapRange_comp] <;>
          /-
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝² : (i : ι) → AddZeroClass (β i)
            inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
            inst✝ : (i : ι) → AddZeroClass (β₂ i)
            e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
            x : DFinsupp fun i => β₁ i
            ⊢ Eq (DFinsupp.mapRange (fun i => Function.comp (fun x => (e i).symm x) fun x  …
          -/
          /-
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝² : (i : ι) → AddZeroClass (β i)
            inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
            inst✝ : (i : ι) → AddZeroClass (β₂ i)
            e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
            x : DFinsupp fun i => β₁ i
            ⊢ Eq (DFinsupp.mapRange (fun i => id) ⋯ x) x
          -/
          /-
            🎉 no goals
          -/
          /-
            case h
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝² : (i : ι) → AddZeroClass (β i)
            inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
            inst✝ : (i : ι) → AddZeroClass (β₂ i)
            e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
            x : DFinsupp fun i => β₁ i
            ⊢ ∀ (i : ι), Eq (id 0) 0
          -/
          simp
          /-
            🎉 no goals
          -/
    right_inv := fun x => by
      /-
        ι : Type u
        γ : Type w
        β : ι → Type v
        β₁ : ι → Type v₁
        β₂ : ι → Type v₂
        inst✝² : (i : ι) → AddZeroClass (β i)
        inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
        inst✝ : (i : ι) → AddZeroClass (β₂ i)
        e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
        x : DFinsupp fun i => β₂ i
        ⊢ Eq (DFinsupp.mapRange (fun i x => (e i) x) ⋯ (DFinsupp.mapRange (fun i x =>  …
      -/
      rw [← mapRange_comp] <;>
          /-
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝² : (i : ι) → AddZeroClass (β i)
            inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
            inst✝ : (i : ι) → AddZeroClass (β₂ i)
            e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
            x : DFinsupp fun i => β₂ i
            ⊢ Eq (DFinsupp.mapRange (fun i => Function.comp (fun x => (e i) x) fun x => (e …
          -/
          /-
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝² : (i : ι) → AddZeroClass (β i)
            inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
            inst✝ : (i : ι) → AddZeroClass (β₂ i)
            e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
            x : DFinsupp fun i => β₂ i
            ⊢ Eq (DFinsupp.mapRange (fun i => id) ⋯ x) x
          -/
          /-
            🎉 no goals
          -/
          /-
            case h
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            inst✝² : (i : ι) → AddZeroClass (β i)
            inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
            inst✝ : (i : ι) → AddZeroClass (β₂ i)
            e : (i : ι) → AddEquiv (β₁ i) (β₂ i)
            x : DFinsupp fun i => β₂ i
            ⊢ ∀ (i : ι), Eq (id 0) 0
          -/
          simp }
          /-
            🎉 no goals
          -/


@[simp]
theorem mapRange.addEquiv_refl :
    (mapRange.addEquiv fun i => AddEquiv.refl (β₁ i)) = AddEquiv.refl _ :=
  AddEquiv.ext mapRange_id


theorem mapRange.addEquiv_trans (f : ∀ i, β i ≃+ β₁ i) (f₂ : ∀ i, β₁ i ≃+ β₂ i) :
    (mapRange.addEquiv fun i => (f i).trans (f₂ i)) =
      (mapRange.addEquiv f).trans (mapRange.addEquiv f₂) := by
  /-
    ι : Type u
    β : ι → Type v
    β₁ : ι → Type v₁
    β₂ : ι → Type v₂
    inst✝² : (i : ι) → AddZeroClass (β i)
    inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
    inst✝ : (i : ι) → AddZeroClass (β₂ i)
    f : (i : ι) → AddEquiv (β i) (β₁ i)
    f₂ : (i : ι) → AddEquiv (β₁ i) (β₂ i)
    ⊢ Eq (DFinsupp.mapRange.addEquiv fun i => (f i).trans (f₂ i)) ((DFinsupp.mapRa …
  -/
  refine AddEquiv.ext <| mapRange_comp (fun i x => f₂ i x) (fun i x => f i x) ?_ ?_ ?_
    /-
      case refine_1
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
      inst✝ : (i : ι) → AddZeroClass (β₂ i)
      f : (i : ι) → AddEquiv (β i) (β₁ i)
      f₂ : (i : ι) → AddEquiv (β₁ i) (β₂ i)
      ⊢ ∀ (i : ι), Eq ((fun i x => (f₂ i) x) i 0) 0
    -/
  · intros; apply map_zero
            /-
              🎉 no goals
            -/
    /-
      case refine_2
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
      inst✝ : (i : ι) → AddZeroClass (β₂ i)
      f : (i : ι) → AddEquiv (β i) (β₁ i)
      f₂ : (i : ι) → AddEquiv (β₁ i) (β₂ i)
      ⊢ ∀ (i : ι), Eq ((fun i x => (f i) x) i 0) 0
    -/
  · intros; apply map_zero
            /-
              🎉 no goals
            -/
    /-
      case refine_3
      ι : Type u
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      inst✝² : (i : ι) → AddZeroClass (β i)
      inst✝¹ : (i : ι) → AddZeroClass (β₁ i)
      inst✝ : (i : ι) → AddZeroClass (β₂ i)
      f : (i : ι) → AddEquiv (β i) (β₁ i)
      f₂ : (i : ι) → AddEquiv (β₁ i) (β₂ i)
      ⊢ ∀ (i : ι), Eq (Function.comp ((fun i x => (f₂ i) x) i) ((fun i x => (f i) x) …
    -/
  · intros; dsimp; simp only [map_zero]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mapRange.addEquiv_symm (e : ∀ i, β₁ i ≃+ β₂ i) :
    (mapRange.addEquiv e).symm = mapRange.addEquiv fun i => (e i).symm :=
  rfl


